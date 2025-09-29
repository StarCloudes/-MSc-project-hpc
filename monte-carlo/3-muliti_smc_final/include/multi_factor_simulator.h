#pragma once
#include "debtor.h"
#include "default_event.h" // 引入新的头文件
#include <vector>

// 多因子模型下单次路径损失模拟函数
// 返回值从 double 升级为 std::vector<DefaultEvent>
std::vector<DefaultEvent> simulate_single_path_multifactor(
    const std::vector<Debtor>& portfolio,
    const std::vector<double>& correlated_factors, // K个相关的系统因子
    const std::vector<double>& epsilon_factors     // N个独立的异质性因子
);
