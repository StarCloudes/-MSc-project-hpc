// monte_carlo_final.cpp
// ============================================================================
// Monte Carlo credit risk simulator (主程序 / main program)
// 模式 / Modes: GAN | Gaussian Copula | Student-t Copula
//
// Stage 1: Preprocessing (Rank 0 上执行)
//    Controller (monte_carlo_final.c) → 总控程序，启动整个流程。
//    Portfolio Generator → 生成债务人组合（敞口、PD、评级、因子载荷）。
//    Select Risk Model
//
// Stage 2: Data Distribution (MPI Broadcast/Scatter)
//    用 MPI 通信把数据分发给所有计算节点：
//    Broadcast → 所有 rank 共享的数据（如矩阵 𝐿）。
//    Scatter → 把组合/任务划分后分发给不同 rank。
//
// Stage 3: Parallel Simulation (All Ranks 并行执行)
//    每个 MPI 进程管理一部分任务 (M_local)。
//    在进程内用 OpenMP 线程并行执行多个路径。
//    模拟过程：
//        Risk Factor Generator → 生成系统性和个体因子（高斯/Student-t/GAN）。
//        Core Simulator (multi_factor_simulator) → 根据因子 + 组合，模拟违约，计算损失。
//    输出：每个线程本地的损失结果 (Thread-Local Loss)。
//
// Stage 4: Results Reduction (归并)
//     Thread-level Reduction → 先把同一进程里不同线程的结果合并。
//     Process-level Reduction (MPI) → 再把不同进程的结果汇总成一个全局直方图 (Global Loss Histogram)。
//
// Stage 5: Analysis & Reporting (Rank 0 上执行)
//     用全局直方图，计算风险指标： Cross-rank Allreduce to aggregate histograms
//         VaR (Value at Risk)
//         ES (Expected Shortfall)
//     输出最终结果：Final Risk（最终风险度量）
// ============================================================================

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <map>
#include <algorithm>
#include <iomanip>

#include "debtor.h"
#include "portfolio_generator.h"
#include "histogram.h"
#include "broadcast_utils.h"
#include "cholesky.h"
#include "multi_factor_simulator.h"
#include "calibrated_parameters.h"
#include "default_event.h"
#include "gan_preprocess.h"  

// ------------------------- 可配置开关 / Feature Toggles -----------------------
// 对 GAN 场景列执行均值为 0、方差为 1 的标准化
// Standardize GAN columns to zero mean and unit variance
static constexpr bool GAN_STANDARDIZE            = true;

// 若提供了真实参考 CSV，则做“分位数映射”：将 GAN 边际映射到参考边际
// If a real reference CSV is provided, perform per-column quantile mapping
static constexpr bool GAN_ENABLE_QUANTILE_MAP    = true;

// Iman–Conover 重排：在保留边际的前提下，使秩相关逼近目标 Σ
// Iman–Conover rank reordering to match target correlation structure (Σ)
static constexpr bool GAN_ENABLE_IMAN_CONOVER    = true;

// 白化→着色微调 Pearson 相关（通常仅在无 real_ref 时启用）
// Whiten-then-colorize to pin Pearson correlations to Σ (often only when no real_ref)
static constexpr bool GAN_FINE_TUNE_WHITEN_COLOR = false;

// t 分布是否缩放到单位方差（便于与 Gaussian 对齐比较）
// Whether to rescale Student-t samples to unit variance for fair comparison
static constexpr bool T_STUDENT_UNIT_VARIANCE    = true;

// 是否打印因子诊断（mean/std/corr），以及抽样行数
// Whether to print factor diagnostics and the sample size for it
static constexpr bool PRINT_FACTOR_DIAGNOSTICS   = true;
static constexpr int  DIAG_SAMPLES               = 50000;

// 模式枚举 / Mode enum
enum class Mode { GAN, GAUSSIAN, STUDENT_T };

int main(int argc, char** argv) {
    // --------------------- MPI 初始化 / Initialize MPI -----------------------
    MPI_Init(&argc, &argv);
    int world_size=1, world_rank=0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 整体 wall-clock 起点 / Global wall-clock start
    const double wall_start = MPI_Wtime();

    // --------------------- 参数解析 / Parse CLI Arguments -------------------
    // 默认模式：Gaussian；若选择 student-t 可传入自由度 nu；GAN 可传 CSV 路径
    // Defaults to Gaussian; student-t accepts <nu>; GAN accepts <gan_csv> [real_ref_csv]
    Mode mode = Mode::GAUSSIAN;
    double dof = 5.0;                       // Student-t degrees of freedom
    std::string gan_file = "gan_scenarios.csv";
    std::string real_ref_file = "";         // optional reference CSV for quantile mapping

    if (argc > 1) {
        std::string m = argv[1];
        if (m=="gan") {
            mode = Mode::GAN;
            if (argc > 2) gan_file = argv[2];
            if (argc > 3) real_ref_file = argv[3];
        } else if (m=="student-t") {
            mode = Mode::STUDENT_T;
            if (argc > 2) dof = std::stod(argv[2]);
        } else if (m=="gaussian") {
            mode = Mode::GAUSSIAN;
        } else {
            if (world_rank==0)
                std::cerr << "usage: " << argv[0]
                          << " [gan <gan_csv> [real_ref_csv] | gaussian | student-t <nu>]\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // --------------------- 规模设定 / Problem Size --------------------------
    // N: 债务人个数；M_total: 总模拟路径数（按 MPI 进程均分，一个进程再用 OpenMP 并行）
    // N: number of debtors; M_total: total paths (partitioned across MPI ranks; OMP within rank)
    const int N = 100000;
    const int M_total = 1000000;

    // 计算每个 rank 分得的行数（支持不能整除的情况）/ Scatter row counts (handles remainders)
    std::vector<int> counts_rows, displs_rows;
    mcutil::partition_rows(world_size, M_total, counts_rows, displs_rows);
    const int M_local = counts_rows[world_rank];

    // 读取（或构造）校准参数（LGD 分布等）
    // Pull calibrated parameters (e.g., LGD distributions by state/rating)
    auto params = get_calibrated_parameters();

    // ============================ I/O 阶段 / I/O Stage =======================
    double t_io_start = MPI_Wtime();

    std::vector<Debtor> portfolio;                 // 投资组合 / portfolio of debtors
    std::vector<std::vector<double>> L_factor;     // 目标相关矩阵的 Cholesky / Cholesky of target Σ

    if (world_rank==0) {
        // 1) 构建合成投资组合（或接入你已有的真实组合生成器）
        // 1) Build a synthetic portfolio (or plug your real portfolio generator here)
        portfolio = generate_synthetic_portfolio(N);

        // 2) 设定目标相关矩阵 Σ，并做 Cholesky 分解，后续用于
        //    - Gaussian / t 采样的“着色”
        //    - GAN 模式下 Iman–Conover / 白化→着色 的目标
        // 2) Build target correlation Σ; Cholesky used by Gaussian/t sampling & GAN calibration
        std::vector<std::vector<double>> factor_corr(NUM_FACTORS, std::vector<double>(NUM_FACTORS, 0.7));
        for (int i=0;i<NUM_FACTORS;++i) factor_corr[i][i]=1.0;
        if (!cholesky_decompose(factor_corr, L_factor)) {
            std::cerr << "错误：目标相关矩阵 Cholesky 失败 / Cholesky failed for target Σ\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 仅在 GAN 模式下读入并预处理所有场景（先在 rank0 做完，再 Scatterv 分发）
    // Read & preprocess GAN scenarios only on rank 0, then Scatterv to workers
    int gan_rows=0, gan_cols=NUM_FACTORS;
    std::vector<double> gan_all_flat, real_ref_flat; int real_rows=0;

    if (mode==Mode::GAN && world_rank==0) {
        // 2.1 读取 GAN CSV（行=路径，列=因子）；不足则报错
        //     Load GAN CSV; fail if fewer rows than M_total
        auto gan_rows2d = mcutil::read_matrix_csv(gan_file, NUM_FACTORS);
        if ((int)gan_rows2d.size() < M_total) {
            std::cerr << "错误：GAN 场景不足，需 " << M_total << "，实际 " << gan_rows2d.size() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        gan_rows2d.resize(M_total);
        gan_rows = M_total;
        gan_all_flat = mcutil::flatten_row_major(gan_rows2d);

        // 2.2 可选：读取真实参考 CSV，用于分位数映射（对齐边际分布）
        //     Optionally read a real reference CSV for quantile mapping (align marginals)
        if (!real_ref_file.empty()) {
            auto real2d = mcutil::read_matrix_csv(real_ref_file, NUM_FACTORS);
            if (!real2d.empty()) { real_rows=(int)real2d.size(); real_ref_flat=mcutil::flatten_row_major(real2d); }
            else std::cerr << "警告：真实参考数据读取失败，将跳过分位数映射 / real_ref read failed; skip quantile mapping\n";
        }

        // 2.3 预处理流水线 / Post-processing pipeline on rank 0
        if (GAN_STANDARDIZE)
            mcutil::standardize_columns_inplace(gan_all_flat, gan_rows, gan_cols); // 标准化 / standardize
        if (GAN_ENABLE_QUANTILE_MAP && real_rows>0)
            mcutil::quantile_map_inplace(gan_all_flat, gan_rows, gan_cols, real_ref_flat, real_rows); // 分位数映射 / quantile map
        if (GAN_ENABLE_IMAN_CONOVER)
            mcutil::iman_conover_reorder_inplace(gan_all_flat, gan_rows, gan_cols, L_factor);        // Iman–Conover 重排
        if (GAN_FINE_TUNE_WHITEN_COLOR)
            mcutil::whiten_then_colorize_inplace(gan_all_flat, gan_rows, gan_cols, L_factor);        // 白化→着色微调

        // 2.4 诊断：打印 mean/std/corr（抽样若干行）
        //     Diagnostics: print mean/std/corr on a sample of rows
        if (PRINT_FACTOR_DIAGNOSTICS) {
            std::string tag = std::string("GAN after preprocess (")
                + (GAN_ENABLE_QUANTILE_MAP && real_rows>0 ? "QuantileMap," : "")
                + (GAN_ENABLE_IMAN_CONOVER ? "IC," : "")
                + (GAN_FINE_TUNE_WHITEN_COLOR ? "WhitenColor" : "") + ")";
            mcutil::print_factor_diag_from_flat(gan_all_flat, gan_rows, gan_cols, DIAG_SAMPLES, tag.c_str());
        }
    }

    // 对照模式（Gaussian / Student-t）也打印一次诊断（用合成样本）
    // For baselines (Gaussian / t), also print diagnostics from synthetic samples
    if (world_rank==0 && mode!=Mode::GAN && PRINT_FACTOR_DIAGNOSTICS) {
        std::mt19937 g(123456);
        int n=DIAG_SAMPLES, K=NUM_FACTORS;
        std::vector<double> flat((size_t)n*K);
        if (mode==Mode::GAUSSIAN) {
            std::normal_distribution<double> nd(0.0,1.0);
            for (int r=0;r<n;++r){
                std::vector<double> u(K); for (int k=0;k<K;++k) u[k]=nd(g);
                for (int i=0;i<K;++i){ double acc=0.0; for (int j=0;j<K;++j) acc+=L_factor[i][j]*u[j]; flat[(size_t)r*K+i]=acc; }
            }
            mcutil::print_factor_diag_from_flat(flat, n, K, n, "Gaussian (synthetic sample)");
        } else { // STUDENT_T
            for (int r=0;r<n;++r){
                auto v = mcutil::generate_correlated_student_t(L_factor, dof, g);
                for (int i=0;i<K;++i) flat[(size_t)r*K + i]=v[i];
            }
            mcutil::print_factor_diag_from_flat(flat, n, K, n, "Student-t (synthetic sample)");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_io_end = MPI_Wtime();

    // ========================= 分发阶段 / Distribution =======================
    double t_dist_start = MPI_Wtime();

    // 广播 portfolio 与 L（所有 rank 都需要）/ Broadcast portfolio and L to all ranks
    broadcast_portfolio_multifactor(portfolio, NUM_FACTORS, 0);
    broadcast_matrix(L_factor, 0);

    // 将 GAN 场景行按 Scatterv 切给各 rank（行不均匀也可）
    // Scatter GAN scenario rows to ranks (supports uneven row counts)
    std::vector<double> gan_local_flat; // [M_local x NUM_FACTORS]
    if (mode==Mode::GAN) {
        MPI_Bcast(&gan_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&gan_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<int> counts_e(world_size), displs_e(world_size);
        for (int r=0;r<world_size;++r){ counts_e[r]=counts_rows[r]*gan_cols; displs_e[r]=displs_rows[r]*gan_cols; }
        gan_local_flat.resize((size_t)M_local*gan_cols);
        MPI_Scatterv(world_rank==0?gan_all_flat.data():nullptr, counts_e.data(), displs_e.data(), MPI_DOUBLE,
                     gan_local_flat.data(), counts_e[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_dist_end = MPI_Wtime();

    // ---------------------- 运行信息打印 / Run Header ------------------------
    if (world_rank==0) {
        std::cout << "\n🚀 开始进行信贷组合风险模拟...\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "投资组合规模 (N)  : " << N << "\n";
        std::cout << "总模拟路径数 (M)    : " << M_total << "\n";
        std::cout << "MPI并行进程数      : " << world_size << "\n";
        std::cout << "每个进程的OMP线程数: " << omp_get_max_threads() << "\n";
        if (mode==Mode::GAN) {
            std::cout << "风险因子来源       : GAN 生成 (" << gan_file << ")\n";
            std::cout << "GAN 标准化         : " << (GAN_STANDARDIZE ? "是":"否") << "\n";
            std::cout << "GAN 分位数映射     : " << ((GAN_ENABLE_QUANTILE_MAP && !real_ref_file.empty()) ? "是":"否") << "\n";
            std::cout << "GAN Iman–Conover   : " << (GAN_ENABLE_IMAN_CONOVER ? "是":"否") << "\n";
            std::cout << "GAN 白化→着色微调  : " << (GAN_FINE_TUNE_WHITEN_COLOR ? "是":"否") << "\n";
            if (!real_ref_file.empty()) std::cout << "真实参考数据       : " << real_ref_file << "\n";
        } else if (mode==Mode::STUDENT_T) {
            std::cout << "风险因子分布       : 学生t分布 (自由度=" << dof
                      << ", 单位方差=" << (T_STUDENT_UNIT_VARIANCE?"是":"否") << ")\n";
        } else {
            std::cout << "风险因子分布       : 正态分布 (高斯Copula)\n";
        }
        std::cout << "--------------------------------------------------\n";
    }

    // =========================== 仿真阶段 / Simulation =======================
    double t_sim_start = MPI_Wtime();

    // 各 rank 的局部直方图（总损失 & 行业损失），线程间采用“本地累加 + 临界区归并”
    // Per-rank local histograms; per-thread accumulates then merges in a critical section
    Histogram local_total_loss_hist_main(0.0, 6e9, 2000);
    std::map<int, Histogram> local_industry_loss_hists_main;
    for (int k=0;k<NUM_FACTORS;++k)
        local_industry_loss_hists_main.emplace(k, Histogram(0.0, 1e9, 500));

    #pragma omp parallel
    {
        Histogram total_hist_thr(0.0, 6e9, 2000);
        std::map<int, Histogram> ind_hist_thr;
        for (int k=0;k<NUM_FACTORS;++k) ind_hist_thr.emplace(k, Histogram(0.0, 1e9, 500));

        // 每线程独立 RNG，避免竞争与相关 / Per-thread RNG for independence
        std::mt19937 gen_thr((unsigned)world_rank * 1000003u + (unsigned)omp_get_thread_num()*101u + 42u);
        std::normal_distribution<double> nd(0.0,1.0);
        std::uniform_real_distribution<double> uni01(0.0,1.0);

        #pragma omp for
        for (int i=0;i<M_local;++i) {
            // 1) 抽取市场状态（此处简单地 5% 压力；如需马尔可夫转移可改用 params.transition_matrix）
            //    Sample market state (here: 5% stressed; plug Markov transition if needed)
            MarketState state = (uni01(gen_thr) < 0.05) ? MarketState::STRESSED : MarketState::QUIET;

            // 2) 按状态/评级抽样 LGD（并裁剪到 [0,1]）
            //    Sample LGD by (state, rating), then clamp to [0,1]
            std::vector<Debtor> port_th = portfolio;
            for (auto& d: port_th) {
                auto p = params.lgd_dists[state][d.rating];
                d.conditional_lgd = std::normal_distribution<>(p.first, p.second)(gen_thr);
                d.conditional_lgd = std::max(0.0, std::min(1.0, d.conditional_lgd));
            }

            // 3) 系统因子：GAN / Student-t / Gaussian（三选一）
            //    Systemic factors per path: GAN / Student-t / Gaussian
            std::vector<double> factors;
            if (mode==Mode::GAN) {
                factors.resize(NUM_FACTORS);
                const double* row=&gan_local_flat[(size_t)i*NUM_FACTORS];
                std::copy(row, row+NUM_FACTORS, factors.begin());
            } else if (mode==Mode::STUDENT_T) {
                factors = mcutil::generate_correlated_student_t(L_factor, dof, gen_thr);
            } else {
                std::vector<double> z(NUM_FACTORS);
                for (int k=0;k<NUM_FACTORS;++k) z[k]=nd(gen_thr);
                factors = generate_correlated_vector(L_factor, z);
            }

            // 4) 特异因子 / Idiosyncratic factors
            std::vector<double> eps(N);
            for (int j=0;j<N;++j) eps[j]=nd(gen_thr);

            // 5) 运行单路径仿真，返回违约事件清单
            //    Simulate one path, get default events
            auto defaults = simulate_single_path_multifactor(port_th, factors, eps);

            // 6) 汇总总损失与行业损失并入线程直方图
            //    Aggregate total & per-industry losses into thread-local histograms
            double total=0.0; std::map<int,double> by_ind;
            for (const auto& ev: defaults){ total+=ev.loss; by_ind[ev.industry_id]+=ev.loss; }
            total_hist_thr.add(total);
            for (const auto& kv: by_ind) ind_hist_thr.at(kv.first).add(kv.second);
        }

        // 7) 线程→进程归并 / Merge thread histograms into per-rank histograms
        #pragma omp critical
        {
            for (size_t b=0; b<local_total_loss_hist_main.data().size(); ++b)
                local_total_loss_hist_main.data()[b] += total_hist_thr.data()[b];
            for (const auto& kv: ind_hist_thr) {
                const int key=kv.first; const auto& ht=kv.second;
                for (size_t b=0; b<ht.data().size(); ++b)
                    local_industry_loss_hists_main.at(key).data()[b] += ht.data()[b];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_sim_end = MPI_Wtime();

    // =========================== 归并阶段 / Reduction =========================
    double t_red_start = MPI_Wtime();

    // 进程间 Allreduce 聚合直方图（总损失 & 各行业）
    // Cross-rank Allreduce to aggregate histograms (total & per-industry)
    std::vector<int> global_total_bins(2000);
    MPI_Allreduce(local_total_loss_hist_main.data().data(), global_total_bins.data(),
                  2000, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::map<int, std::vector<int>> global_industry_bins;
    for (int k=0;k<NUM_FACTORS;++k) {
        global_industry_bins[k].resize(500);
        MPI_Allreduce(local_industry_loss_hists_main.at(k).data().data(),
                      global_industry_bins[k].data(), 500, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_red_end = MPI_Wtime();

    // =========================== 结果输出 / Reporting =========================
    if (world_rank==0) {
        Histogram final_total_hist(0.0, 6e9, 2000);
        final_total_hist.set_data(global_total_bins);

        // 同时报出 VaR/ES 的 95% 与 99%
        // Report both 95% and 99% VaR/ES
        double var95 = final_total_hist.estimate_var(0.95);
        double es95  = final_total_hist.estimate_es(0.95);
        double var99 = final_total_hist.estimate_var(0.99);
        double es99  = final_total_hist.estimate_es(0.99);

        std::cout << "\n✅ Monte Carlo simulation completed\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "总体风险指标 / Portfolio Risk Metrics\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << "Total paths      : " << M_total << "\n";
        std::cout << "VaR (95%)        : " << var95 << "\n";
        std::cout << "ES  (95%)        : " << es95  << "\n";
        std::cout << "VaR (99%)        : " << var99 << "\n";
        std::cout << "ES  (99%)        : " << es99  << "\n";

        // 行业分解：打印 ES95 与 ES99，按 ES99 排序，便于与历史口径比对
        // Industry breakdown: print ES95 & ES99; sort by ES99 for consistent ranking
        std::cout << "--------------------------------------------------\n";
        std::cout << "按行业划分的风险贡献 (Top 5 by ES99) / Industry ES\n";
        std::cout << "--------------------------------------------------\n";

        std::vector<std::tuple<int,double,double>> industry_es; // (id, ES95, ES99)
        industry_es.reserve(NUM_FACTORS);
        for (int k=0;k<NUM_FACTORS;++k) {
            Histogram h(0.0, 1e9, 500);
            h.set_data(global_industry_bins[k]);
            industry_es.emplace_back(k, h.estimate_es(0.95), h.estimate_es(0.99));
        }
        std::sort(industry_es.begin(), industry_es.end(),
                  [](const auto& a, const auto& b){ return std::get<2>(a) > std::get<2>(b); });

        for (int i=0; i<5 && i<(int)industry_es.size(); ++i) {
            int id = std::get<0>(industry_es[i]);
            double es95k = std::get<1>(industry_es[i]);
            double es99k = std::get<2>(industry_es[i]);
            std::cout << "行业 " << std::setw(2) << id
                      << " | ES(95%): " << std::fixed << std::setprecision(2) << es95k
                      << " | ES(99%): " << std::fixed << std::setprecision(2) << es99k << "\n";
        }
        std::cout << "--------------------------------------------------\n";
    }

    // ============================= 计时 / Timing =============================
    MPI_Barrier(MPI_COMM_WORLD);
    const double wall_end = MPI_Wtime();
    if (world_rank==0) {
        double t_io   = t_io_end   - t_io_start;
        double t_dist = t_dist_end - t_dist_start;
        double t_sim  = t_sim_end  - t_sim_start;
        double t_red  = t_red_end  - t_red_start;
        double t_tot  = wall_end   - wall_start;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n⏱️ 阶段计时 / Stage timing (seconds)\n";
        std::cout << "I/O               : " << t_io   << "\n";
        std::cout << "分发 / Distribution: " << t_dist << "\n";
        std::cout << "仿真 / Simulation  : " << t_sim  << "\n";
        std::cout << "归并 / Reduction   : " << t_red  << "\n";
        std::cout << "-------------------------------------\n";
        std::cout << "总运行时间 (Wall-clock): " << t_tot << " 秒\n";
    }

    // ------------------------------ 收尾 / Teardown --------------------------
    MPI_Finalize();
    return 0;
}
