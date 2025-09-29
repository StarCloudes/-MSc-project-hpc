#pragma once

#include <vector>
#include <map>

// 定义市场状态
enum class MarketState {
    QUIET,
    STRESSED
};

// 定义信用评级
enum class CreditRating {
    AAA, AA, A, BBB, BB, B, CCC
};

// 存储校准后的状态依赖参数
struct CalibratedStateParams {
    // 状态转移概率矩阵
    // P[i][j] 是从状态 i 转移到状态 j 的概率
    std::map<MarketState, std::map<MarketState, double>> transition_matrix;

    // 不同状态下，各类信用评级对应的因子载荷分布
    // (均值, 标准差)
    std::map<MarketState, std::map<CreditRating, std::pair<double, double>>> factor_loading_dists;
    
    // 不同状态下，各类信用评级对应的LGD分布
    // (均值, 标准差)
    std::map<MarketState, std::map<CreditRating, std::pair<double, double>>> lgd_dists;
};

// 全局函数，返回一组“伪”校准出的参数
// 在真实世界中，这些值将从一个外部文件（如JSON或数据库）中读取
inline CalibratedStateParams get_calibrated_parameters() {
    CalibratedStateParams params;

    // 示例：状态转移概率 (数值为示例)
    params.transition_matrix[MarketState::QUIET][MarketState::QUIET] = 0.95;
    params.transition_matrix[MarketState::QUIET][MarketState::STRESSED] = 0.05;
    params.transition_matrix[MarketState::STRESSED][MarketState::STRESSED] = 0.80;
    params.transition_matrix[MarketState::STRESSED][MarketState::QUIET] = 0.20;

    // 示例：因子载荷分布 (均值, 标准差)
    // 压力状态下，因子载荷的均值和波动都更大
    params.factor_loading_dists[MarketState::QUIET][CreditRating::AAA] = {0.1, 0.05};
    params.factor_loading_dists[MarketState::QUIET][CreditRating::BBB] = {0.3, 0.10};
    params.factor_loading_dists[MarketState::QUIET][CreditRating::B]   = {0.5, 0.15};
    params.factor_loading_dists[MarketState::STRESSED][CreditRating::AAA] = {0.2, 0.10};
    params.factor_loading_dists[MarketState::STRESSED][CreditRating::BBB] = {0.5, 0.15};
    params.factor_loading_dists[MarketState::STRESSED][CreditRating::B]   = {0.7, 0.20};

    // 示例：LGD分布 (均值, 标准差)
    // 压力状态下，LGD更高
    params.lgd_dists[MarketState::QUIET][CreditRating::AAA] = {0.20, 0.05};
    params.lgd_dists[MarketState::QUIET][CreditRating::BBB] = {0.40, 0.10};
    params.lgd_dists[MarketState::QUIET][CreditRating::B]   = {0.60, 0.10};
    params.lgd_dists[MarketState::STRESSED][CreditRating::AAA] = {0.30, 0.10};
    params.lgd_dists[MarketState::STRESSED][CreditRating::BBB] = {0.55, 0.15};
    params.lgd_dists[MarketState::STRESSED][CreditRating::B]   = {0.75, 0.15};

    return params;
}
