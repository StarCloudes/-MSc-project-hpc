#pragma once

#include <vector>
#include <random>

// Cholesky分解函数，将一个对称正定矩阵 A 分解为 A = L * L^T
bool cholesky_decompose(
    const std::vector<std::vector<double>>& A,
    std::vector<std::vector<double>>& L
);

// 使用Cholesky下三角矩阵 L 将一组独立随机向量转换为相关随机向量
std::vector<double> generate_correlated_vector(
    const std::vector<std::vector<double>>& L,
    const std::vector<double>& independent_vector
);
