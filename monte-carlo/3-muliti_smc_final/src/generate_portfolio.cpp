// This function builds a realistic synthetic portfolio for testing risk models.
// It balances structure (industry factor loadings, calibrated parameters) 
// with randomness (exposures, PDs, noise).

#include "portfolio_generator.h"
#include "calibrated_parameters.h"
#include <random>
#include <cmath>
#include <numeric>

// 从 portfolio_generator.h 引入因子数量的定义
extern const int NUM_FACTORS;

/**
 * @brief 辅助函数：根据预设概率分布随机选择一个信用评级
 *        Helper function: randomly select a credit rating based on preset probabilities
 * 
 * 使用 std::discrete_distribution 按权重抽取评级，权重可调整以反映实际分布。
 * Uses std::discrete_distribution to sample ratings according to weights.
 * 
 * @param gen 随机数生成器 / Random number generator
 * @return CreditRating 随机选中的信用评级 / Randomly selected credit rating
 */
CreditRating get_random_rating(std::mt19937& gen) {
    // 权重顺序对应于 CreditRating 枚举顺序
    // Weights correspond to CreditRating enum order
    std::discrete_distribution<> dist({
        10.0, // AAA
        15.0, // AA
        20.0, // A
        25.0, // BBB
        15.0, // BB
        10.0, // B
        5.0   // CCC
    });
    int index = dist(gen); // 按权重抽取索引 / Sample index by weights
    return static_cast<CreditRating>(index); // 转换为枚举类型 / Cast to enum type
}

/**
 * @brief 生成一个合成债务人投资组合
 *        Generate a synthetic portfolio of debtors
 * 
 * 每个债务人分配有唯一ID、风险敞口、违约概率、信用评级、行业ID、因子载荷和R方。
 * Each debtor is assigned a unique ID, exposure, PD, credit rating, industry ID, factor loadings, and R-squared.
 * 
 * @param N 投资组合规模 / Portfolio size
 * @return std::vector<Debtor> 债务人列表 / List of debtors
 */
std::vector<Debtor> generate_synthetic_portfolio(int N) {
    std::vector<Debtor> portfolio;
    portfolio.reserve(N); // 预分配空间 / Reserve space for efficiency

    // 使用固定种子的高质量随机数引擎，保证结果可复现
    // Use fixed-seed mt19937 for reproducibility
    std::mt19937 gen(42); 

    // 违约概率分布（0.1% ~ 25%）
    // PD distribution (0.1% ~ 25%)
    std::uniform_real_distribution<double> pd_dist(0.001, 0.25);

    // 风险敞口分布（1万 ~ 10万）
    // Exposure distribution (10,000 ~ 100,000)
    std::uniform_real_distribution<double> exp_dist(10000.0, 100000.0);

    // 行业ID分布（0 ~ NUM_FACTORS-1）
    // Industry ID distribution (0 ~ NUM_FACTORS-1)
    std::uniform_int_distribution<int> industry_dist(0, NUM_FACTORS - 1);

    // 获取校准参数（如因子载荷分布）
    // Get calibrated parameters (e.g., factor loading distributions)
    auto params = get_calibrated_parameters();

    for (int i = 0; i < N; ++i) {
        Debtor d;
        d.id = i; // 唯一ID / Unique ID
        d.exposure = exp_dist(gen); // 风险敞口 / Exposure
        d.pd = pd_dist(gen); // 违约概率 / Probability of default
        d.rating = get_random_rating(gen); // 随机信用评级 / Random credit rating
        d.industry_id = industry_dist(gen); // 随机行业ID / Random industry ID

        // 获取当前评级在QUIET市场状态下的因子载荷分布参数（均值和标准差）
        // Get factor loading distribution parameters for current rating under QUIET market state
        auto loading_params = params.factor_loading_dists[MarketState::QUIET][d.rating];
        std::normal_distribution<double> loading_dist(loading_params.first, loading_params.second);

        // 初始化因子载荷向量为0
        // Initialize factor loadings vector to zero
        d.factor_loadings.assign(NUM_FACTORS, 0.0);

        // 主行业因子载荷：按分布采样并保证非负
        // Main industry factor loading: sample and ensure non-negative
        d.factor_loadings[d.industry_id] = std::max(0.0, loading_dist(gen));

        // 计算主行业因子载荷的平方和
        // Sum of squares for main industry loading
        double loadings_sum_sq = d.factor_loadings[d.industry_id] * d.factor_loadings[d.industry_id];

        // 其他行业因子载荷：加少量噪声
        // Other industry factor loadings: add small noise
        std::normal_distribution<double> noise_dist(0.0, 0.05);
        for(int k=0; k<NUM_FACTORS; ++k) {
            if (k != d.industry_id) {
                d.factor_loadings[k] = noise_dist(gen); // 采样噪声 / Sample noise
                loadings_sum_sq += d.factor_loadings[k] * d.factor_loadings[k];
            }
        }
        // 计算R方（所有因子载荷的平方和）
        // Calculate R-squared (sum of squares of all factor loadings)
        d.r_squared = loadings_sum_sq;

        // 添加到投资组合
        // Add debtor to portfolio
        portfolio.push_back(d);
    }

    return portfolio;
}