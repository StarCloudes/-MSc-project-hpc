#pragma once

#include <vector>
#include "calibrated_parameters.h" // 引入新的头文件

struct Debtor {
    // 基础参数
    int id;          // 债务人唯一ID
    double exposure; // 风险敞口
    double pd;       // 违约概率

    // --- 分段策略参数 ---
    int industry_id;         // 行业ID
    CreditRating rating;     // 信用评级

    // --- 状态依赖参数 ---
    // 这些参数将在模拟时根据市场状态动态确定
    double conditional_lgd;
    std::vector<double> factor_loadings; // 因子载荷向量
    double r_squared;                    // R方
};
