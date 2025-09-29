// 它在多因子 Merton 模型下，完成一条蒙特卡洛路径的违约判定：对每个债务人构造标准化资产回报
// 

#include "multi_factor_simulator.h"
#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include <numeric>

// 初始化标准正态分布对象
// Initialize standard normal distribution object
static const boost::math::normal_distribution<> standard_normal(0.0, 1.0);

/**
 * @brief 多因子模型下单条路径的违约事件模拟
 *        Simulate default events for a single path under a multi-factor model
 *
 * @param portfolio 债务人投资组合 / Portfolio of debtors
 * @param correlated_factors K个相关系统因子 / K correlated systematic factors  受“大环境”的影响，比如行业景气度、市场波动
 * @param epsilon_factors N个独立异质性因子 / N independent idiosyncratic factors 每个债务人还有自己的“小意外”，管理不善、突发事件
 * @return std::vector<DefaultEvent> 违约事件列表 / List of default events
 */
std::vector<DefaultEvent> simulate_single_path_multifactor(
    const std::vector<Debtor>& portfolio,
    const std::vector<double>& correlated_factors,
    const std::vector<double>& epsilon_factors) {

    std::vector<DefaultEvent> defaulted_debtors; // 存储本路径下所有违约债务人 / Store all defaulted debtors for this path

    // 遍历投资组合中的每个债务人 / Iterate over each debtor in the portfolio
    for (size_t i = 0; i < portfolio.size(); ++i) {
        const Debtor& d = portfolio[i];

        // 1. 计算系统性风险部分
        //    通过因子载荷与系统因子的点积获得 / Dot product of factor loadings and systematic factors
        double systematic_part = std::inner_product(
            d.factor_loadings.begin(), 
            d.factor_loadings.end(), 
            correlated_factors.begin(), 
            0.0
        );

        // 2. 计算异质性风险部分
        //    用 sqrt(1 - r_squared) 缩放独立噪声 / Scale idiosyncratic factor by sqrt(1 - r_squared)
        double idiosyncratic_part = std::sqrt(1.0 - d.r_squared) * epsilon_factors[i];

        // 3. 合成总资产价值
        //    总资产价值 = 系统性部分 + 异质性部分 / Total asset value = systematic + idiosyncratic part
        double asset_value = systematic_part + idiosyncratic_part;

        // 4. 计算违约阈值
        //    使用标准正态分布的分位点函数，将违约概率映射为阈值 / Map PD to threshold using standard normal quantile
        double default_threshold = boost::math::quantile(standard_normal, d.pd);

        // 5. 判断是否违约
        //    如果资产价值低于阈值，则发生违约 / If asset value < threshold, default occurs
        if (asset_value < default_threshold) {
            // --- 核心升级 ---
            // 如果发生违约，则记录详细的违约事件，而不是简单累加损失
            // If default occurs, record detailed default event instead of just accumulating loss
            defaulted_debtors.push_back({
                d.id,                        // 债务人ID / Debtor ID
                d.industry_id,               // 行业ID / Industry ID
                d.rating,                    // 信用评级 / Credit rating
                d.exposure * d.conditional_lgd // 损失 = 敞口 * 条件LGD / Loss = exposure * conditional LGD
            });
        }
    }
    return defaulted_debtors;
}