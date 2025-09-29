#pragma once // 使用 #pragma once 可以防止头文件被重复包含，比 #ifndef 更简洁

#include <vector>
#include <random>

/**
 * @brief 使用 Student-t Copula 生成资产价值向量
 * * 此函数通过多维Student-t分布生成随机数，然后通过一系列变换，
 * 最终得到一组具有Student-t尾部相关性的、服从标准正态分布的资产价值。
 * * @param dim 向量的维度 (例如 N+1)
 * @param degrees_of_freedom Student-t分布的自由度 (nu)
 * @param gen 随机数生成器
 * @return std::vector<double> 最终生成的资产价值向量
 */
std::vector<double> generate_student_t_asset_values(
    int dim,
    double degrees_of_freedom,
    std::mt19937& gen
);
