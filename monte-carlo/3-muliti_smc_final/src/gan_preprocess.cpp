// gan_preprocess.cpp
#include "gan_preprocess.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include "cholesky.h"

namespace mcutil {

    // 读取CSV文件为二维矩阵
    // Read a CSV file into a 2D matrix
    std::vector<std::vector<double>> read_matrix_csv(const std::string &filename, int expected_cols) {
        std::vector<std::vector<double>> rows;
        std::ifstream f(filename);
        if (!f.is_open()) {
            std::cerr << "错误：无法打开文件 " << filename << std::endl; // Error: Cannot open file
            return rows;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string tok;
            std::vector<double> row;
            // 逐个读取逗号分隔的数字 / Read comma-separated numbers
            while (std::getline(ss, tok, ','))
                if (!tok.empty()) row.push_back(std::stod(tok));
            // 检查列数是否匹配 / Check column count
            if ((int)row.size() != expected_cols) {
                std::cerr << "错误：文件 " << filename << " 列数=" << row.size() <<
                    " 与模型要求=" << expected_cols << " 不一致\n"; // Error: Column count mismatch
                rows.clear();
                return rows;
            }
            rows.push_back(std::move(row));
        }
        return rows;
    }

    // 按行主序展开二维矩阵为一维数组
    // Flatten a 2D matrix to a 1D array in row-major order
    std::vector<double> flatten_row_major(const std::vector<std::vector<double>> &a) {
        if (a.empty()) return {};
        int R = (int)a.size(), C = (int)a[0].size();
        std::vector<double> out((size_t)R * C);
        // 依次复制每一行到一维数组 / Copy each row into the 1D array
        for (int r = 0; r < R; ++r) std::copy(a[r].begin(), a[r].end(), out.begin() + (size_t)r * C);
        return out;
    }

    // 将总行数分配给各进程，得到每个进程的行数和偏移
    // Partition total rows among MPI processes, get counts and displacements
    void partition_rows(int world_size, int M_total,
        std::vector<int> &counts_rows, std::vector<int> &displs_rows) {
        counts_rows.assign(world_size, M_total / world_size);
        int rem = M_total % world_size;
        // 前rem个进程多分配一行 / First rem processes get one extra row
        for (int i = 0; i < rem; ++i) counts_rows[i] += 1;
        displs_rows.assign(world_size, 0);
        // 计算每个进程的起始偏移 / Calculate displacement for each process
        for (int i = 1; i < world_size; ++i) displs_rows[i] = displs_rows[i - 1] + counts_rows[i - 1];
    }

    // ---------------- 统计/线代 ----------------
    // 对每一列进行标准化（均值为0，方差为1）
    // Standardize each column (mean=0, std=1)
    void standardize_columns_inplace(std::vector<double> &flat, int rows, int cols) {
        if (rows == 0 || cols == 0) return;
        const double eps = 1e-12;
        for (int c = 0; c < cols; ++c) {
            double sum = 0.0;
            // 计算均值 / Calculate mean
            for (int r = 0; r < rows; ++r) sum += flat[(size_t)r * cols + c];
            double mean = sum / rows;
            double vs = 0.0;
            // 计算方差 / Calculate variance
            for (int r = 0; r < rows; ++r) {
                double z = flat[(size_t)r * cols + c] - mean;
                vs += z * z;
            }
            double var = vs / std::max(1, rows - 1);
            double stdv = std::sqrt(std::max(var, eps));
            // 标准化每个元素 / Standardize each element
            for (int r = 0; r < rows; ++r) {
                double &ref = flat[(size_t)r * cols + c];
                ref = (ref - mean) / stdv;
            }
        }
    }

    // 用Cholesky分解矩阵L对每一行进行着色（相关变换）
    // Colorize each row using Cholesky matrix L (correlation transformation)
    void colorize_rows_inplace(std::vector<double> &flat, int rows, int cols,
        const Mat &L) {
        std::vector<double> tmp(cols);
        for (int r = 0; r < rows; ++r) {
            double *row = &flat[(size_t)r * cols];
            // 对每一行做线性变换 / Apply linear transformation to each row
            for (int i = 0; i < cols; ++i) {
                double acc = 0.0;
                for (int j = 0; j < cols; ++j) acc += L[i][j] * row[j];
                tmp[i] = acc;
            }
            std::copy(tmp.begin(), tmp.end(), row);
        }
    }

    // 从一维数组采样相关矩阵
    // Sample correlation matrix from flat array
    std::vector<std::vector<double>> sample_corr_from_flat(const std::vector<double> &flat, int rows, int K) {
        std::vector<double> mean(K, 0.0);
        // 计算每列均值 / Calculate mean for each column
        for (int r = 0; r < rows; ++r) {
            const double *row = &flat[(size_t)r * K];
            for (int i = 0; i < K; ++i) mean[i] += row[i];
        }
        for (int i = 0; i < K; ++i) mean[i] /= rows;

        std::vector<double> var(K, 0.0);
        // 计算每列方差 / Calculate variance for each column
        for (int r = 0; r < rows; ++r) {
            const double *row = &flat[(size_t)r * K];
            for (int i = 0; i < K; ++i) {
                double z = row[i] - mean[i];
                var[i] += z * z;
            }
        }
        for (int i = 0; i < K; ++i) var[i] /= std::max(1, rows - 1);

        std::vector<std::vector<double>> cov(K, std::vector<double>(K, 0.0));
        // 计算协方差矩阵 / Calculate covariance matrix
        for (int r = 0; r < rows; ++r) {
            const double *row = &flat[(size_t)r * K];
            for (int i = 0; i < K; ++i) {
                double xi = row[i] - mean[i];
                for (int j = i; j < K; ++j) {
                    double xj = row[j] - mean[j];
                    cov[i][j] += xi * xj;
                }
            }
        }
        for (int i = 0; i < K; ++i)
            for (int j = i; j < K; ++j) cov[i][j] /= std::max(1, rows - 1);

        std::vector<std::vector<double>> corr(K, std::vector<double>(K, 0.0));
        // 计算相关系数矩阵 / Calculate correlation matrix
        for (int i = 0; i < K; ++i) {
            double si = std::sqrt(std::max(1e-12, var[i]));
            for (int j = 0; j < K; ++j) {
                double sj = std::sqrt(std::max(1e-12, var[j]));
                double c = (i <= j ? cov[i][j] : cov[j][i]);
                corr[i][j] = corr[j][i] = c / (si * sj);
            }
        }
        return corr;
    }

    // 对相关矩阵做Cholesky分解，必要时加jitter
    // Cholesky decomposition with jitter if needed
    bool cholesky_with_jitter(std::vector<std::vector<double>> M, std::vector<std::vector<double>> &L,
        double jitter, int max_tries) {
        for (int t = 0; t <= max_tries; ++t) {
            std::vector<std::vector<double>> A = M;
            // 每次尝试时对角线加jitter，增强正定性 / Add jitter to diagonal to enhance positive-definiteness
            if (t > 0)
                for (size_t i = 0; i < A.size(); ++i) A[i][i] += jitter * std::pow(10.0, t - 1);
            if (cholesky_decompose(A, L)) return true;
        }
        return false;
    }

    // 前向求解线性方程组Ly=x
    // Forward solve linear system Ly=x
    std::vector<double> forward_solve(const Mat &L,
        const std::vector<double> &x) {
        int K = (int)L.size();
        std::vector<double> y(K, 0.0);
        // 按行递推求解 / Forward substitution
        for (int i = 0; i < K; ++i) {
            double acc = x[i];
            for (int j = 0; j < i; ++j) acc -= L[i][j] * y[j];
            y[i] = acc / L[i][i];
        }
        return y;
    }

    // 白化后再着色
    // Whiten then colorize the data
    void whiten_then_colorize_inplace(std::vector<double> &flat, int M, int K,
        const Mat &L_target) {
        auto R = sample_corr_from_flat(flat, M, K); // 采样当前相关矩阵 / Sample current correlation matrix
        Mat A;
        // 尝试Cholesky分解，失败则跳过 / Try Cholesky decomposition, skip if fails
        if (!cholesky_with_jitter(R, A)) {
            std::cerr << "警告：GAN 样本相关不可分解，跳过白化\n"; // Warning: Cannot decompose, skip whitening
            return;
        }
        // 对每一行做白化 / Whiten each row
        for (int r = 0; r < M; ++r) {
            double *row = &flat[(size_t)r * K];
            std::vector<double> x(row, row + K);
            std::vector<double> xw = forward_solve(A, x); // A*y=x
            for (int i = 0; i < K; ++i) row[i] = xw[i];
        }
        // 再着色为目标相关结构 / Colorize to target correlation structure
        colorize_rows_inplace(flat, M, K, L_target);
    }

    // ---------------- GAN 后处理 ----------------
    // 分位数映射，将GAN样本分布映射到真实分布
    // Quantile mapping: map GAN sample distribution to real distribution
    void quantile_map_inplace(std::vector<double> &gan_flat, int M, int K,
        const std::vector<double> &real_flat, int R) {
        // 提取指定列 / Extract specified column
        auto extract_col = [&](const std::vector<double> &f, int rows, int col) {
            std::vector<double> v(rows);
            for (int r = 0; r < rows; ++r) v[r] = f[(size_t)r * K + col];
            return v;
        };
        // 写回指定列 / Write back to specified column
        auto write_col = [&](std::vector<double> &f, int rows, int col,
            const std::vector<double> &v) {
            for (int r = 0; r < rows; ++r) f[(size_t)r * K + col] = v[r];
        };

        for (int c = 0; c < K; ++c) {
            std::vector<double> gx = extract_col(gan_flat, M, c);
            std::vector<double> rx = extract_col(real_flat, R, c);
            std::vector<double> gs = gx, rs = rx;
            std::sort(gs.begin(), gs.end());
            std::sort(rs.begin(), rs.end());

            std::vector<double> mapped(M);
            // 对每个GAN样本，找到其分位数并映射到真实分布 / For each GAN sample, find its quantile and map to real distribution
            for (int r = 0; r < M; ++r) {
                double x = gx[r];
                auto it = std::lower_bound(gs.begin(), gs.end(), x);
                int rank = (int)std::distance(gs.begin(), it);
                double u = (rank + 0.5) / std::max(1, M);
                u = std::min(1.0 - 1e-12, std::max(1e-12, u));
                double pos = u * (R - 1);
                int i = (int)std::floor(pos);
                double frac = pos - i;
                double xr = (i >= R - 1) ? rs[R - 1] : (rs[i] * (1.0 - frac) + rs[i + 1] * frac);
                mapped[r] = xr;
            }
            write_col(gan_flat, M, c, mapped);
        }
    }

    // 生成相关正态分布样本
    // Generate correlated normal samples
    std::vector<double> generate_correlated_normals(int M, int K,
        const Mat &L) {
        std::mt19937 gen(1234567);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<double> out((size_t)M * K);
        std::vector<double> u(K), z(K);
        // 对每个样本路径，生成相关正态因子 / For each sample path, generate correlated normal factors
        for (int r = 0; r < M; ++r) {
            for (int i = 0; i < K; ++i) u[i] = nd(gen);
            for (int i = 0; i < K; ++i) {
                double acc = 0.0;
                for (int j = 0; j < K; ++j) acc += L[i][j] * u[j];
                z[i] = acc;
            }
            for (int i = 0; i < K; ++i) out[(size_t)r * K + i] = z[i];
        }
        return out;
    }

    // Iman-Conover重排序算法
    // Iman-Conover reordering algorithm
    void iman_conover_reorder_inplace(std::vector<double> &flat, int M, int K,
        const Mat &L_target) {
        std::vector<double> Z = generate_correlated_normals(M, K, L_target);

        auto colvals = [&](const std::vector<double> &f, int rows, int col) {
            std::vector<double> v(rows);
            for (int r = 0; r < rows; ++r) v[r] = f[(size_t)r * K + col];
            return v;
        };
        auto write_col = [&](std::vector<double> &f, int rows, int col,
            const std::vector<double> &v) {
            for (int r = 0; r < rows; ++r) f[(size_t)r * K + col] = v[r];
        };

        for (int c = 0; c < K; ++c) {
            // 按相关正态因子的排序重排GAN样本 / Reorder GAN samples by correlated normal factor order
            std::vector<int> z_order(M);
            std::iota(z_order.begin(), z_order.end(), 0);
            std::sort(z_order.begin(), z_order.end(), [&](int a, int b) {
                return Z[(size_t)a * K + c] < Z[(size_t)b * K + c];
            });

            std::vector<double> x = colvals(flat, M, c);
            std::sort(x.begin(), x.end());

            std::vector<double> newcol(M);
            for (int i = 0; i < M; ++i) {
                int row = z_order[i];
                newcol[row] = x[i];
            }
            write_col(flat, M, c, newcol);
        }
    }

    // ---------------- 采样器 ----------------
    // 生成相关的Student-t分布样本
    // Generate correlated Student-t samples
    std::vector<double> generate_correlated_student_t(const Mat &L, double nu, std::mt19937 &gen) {
        int K = (int)L.size();
        std::normal_distribution<double> nd(0.0, 1.0);
        std::chi_squared_distribution<double> chi2(nu);
        std::vector<double> u(K), z(K), t(K);
        // 生成独立正态因子 / Generate independent normal factors
        for (int k = 0; k < K; ++k) u[k] = nd(gen);
        // 相关变换 / Correlation transformation
        for (int i = 0; i < K; ++i) {
            double acc = 0.0;
            for (int j = 0; j < K; ++j) acc += L[i][j] * u[j];
            z[i] = acc;
        }
        // Student-t缩放 / Student-t scaling
        double s = chi2(gen), scale = std::sqrt(s / nu);
        double unit = (nu > 2.0 ? std::sqrt((nu - 2.0) / nu) : 1.0); // 单位方差缩放 / Unit variance scaling
        for (int i = 0; i < K; ++i) t[i] = (z[i] / scale) * unit;
        return t;
    }

    // ---------------- 诊断 ----------------
    // 打印因子诊断信息（均值、标准差、相关系数）
    // Print factor diagnostics (mean, std, correlation)
    void print_factor_diag_from_flat(const std::vector<double> &flat, int rows, int K,
        int sample_rows,
        const char *title) {
        if (rows == 0 || K == 0) return;
        int n = std::min(rows, sample_rows);
        int stride = std::max(1, rows / n);
        std::vector<double> sum(K, 0.0), sumsq(K, 0.0);
        std::vector<std::vector<double>> cross(K, std::vector<double>(K, 0.0));
        int taken = 0;
        // 采样部分数据用于诊断 / Sample part of data for diagnostics
        for (int r = 0; r < rows && taken < n; r += stride, ++taken) {
            const double *row = &flat[(size_t)r * K];
            for (int i = 0; i < K; ++i) {
                sum[i] += row[i];
                sumsq[i] += row[i] * row[i];
            }
            for (int i = 0; i < K; ++i)
                for (int j = i; j < K; ++j) cross[i][j] += row[i] * row[j];
        }
        std::vector<double> mean(K), stdv(K);
        // 计算均值和标准差 / Calculate mean and standard deviation
        for (int i = 0; i < K; ++i) {
            mean[i] = sum[i] / taken;
            double v = (sumsq[i] / taken) - mean[i] * mean[i];
            v = (v > 0 ? v : 0);
            stdv[i] = std::sqrt(v + 1e-12);
        }
        std::vector<std::vector<double>> corr(K, std::vector<double>(K, 0.0));
        // 计算相关系数矩阵 / Calculate correlation matrix
        for (int i = 0; i < K; ++i) {
            for (int j = i; j < K; ++j) {
                double cij = (cross[i][j] / taken) - mean[i] * mean[j];
                corr[i][j] = corr[j][i] = cij / (stdv[i] * stdv[j] + 1e-12);
            }
        }
        // 打印诊断信息 / Print diagnostics
        std::cout << "\n--- 因子诊断 / Factor Diagnostics (" << title <<
            ", n=" << taken << ", K=" << K << ") ---\n";
        std::cout << "mean: ";
        for (int i = 0; i < K; ++i) std::cout << std::fixed << std::setprecision(4) << mean[i] << (i + 1 < K ? ", " : "\n");
        std::cout << "std : ";
        for (int i = 0; i < K; ++i) std::cout << std::fixed << std::setprecision(4) << stdv[i] << (i + 1 < K ? ", " : "\n");
        std::cout << "corr:\n";
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j)
                std::cout << std::setw(8) << std::setprecision(4) << corr[i][j] << " ";
            std::cout << "\n";
        }
        std::cout << "--------------------------------------------------\n";
    }

} // namespace mcutil