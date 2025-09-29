#pragma once

#include "debtor.h"
#include <vector>

// 广播 K x K 矩阵的函数 (保持不变)
void broadcast_matrix(
    std::vector<std::vector<double>>& matrix, 
    int root_rank
);

// 升级广播函数以支持新的Debtor结构
void broadcast_portfolio_multifactor(
    std::vector<Debtor>& portfolio, 
    int num_factors, 
    int root_rank
);
