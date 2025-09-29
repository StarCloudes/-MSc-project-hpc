

#include "cholesky.h"
#include <cmath>

/**
 * @brief Cholesky分解函数，将对称正定矩阵A分解为A = L * L^T   👉 就像把“大方阵”拆开成“下三角矩阵”和它的转置
 *        Cholesky decomposition: decomposes symmetric positive-definite matrix A into A = L * L^T
 *
 * @param A 输入矩阵 / Input matrix (must be symmetric and positive-definite)
 * @param L 输出下三角矩阵 / Output lower-triangular matrix
 * @return 分解成功返回true，否则返回false / Returns true if decomposition succeeds, false otherwise
 *
 * 算法说明：
 * 对于每个元素L[i][j]，计算公式如下：
 * - 如果i==j（对角线），L[i][j] = sqrt(A[i][j] - sum_k(L[i][k] * L[j][k]))
 * - 如果i>j（下三角），L[i][j] = (A[i][j] - sum_k(L[i][k] * L[j][k])) / L[j][j]
 * 只计算下三角部分，保证L为下三角矩阵。
 */
bool cholesky_decompose(const std::vector<std::vector<double>>& A,
                        std::vector<std::vector<double>>& L) {
    int n = A.size();
    // 初始化L为n x n零矩阵 / Initialize L as n x n zero matrix
    L.assign(n, std::vector<double>(n, 0.0));

    // 遍历每一行
    for (int i = 0; i < n; ++i) {
        // 遍历每一列（只计算下三角部分）
        for (int j = 0; j <= i; ++j) {
            double sum = A[i][j];
            // 累加已知的L元素，计算L[i][j]的剩余部分
            // Accumulate the sum of L[i][k] * L[j][k] for k < j
            for (int k = 0; k < j; ++k) {
                sum -= L[i][k] * L[j][k];
            }
            if (i == j) {
                // 对角线元素，必须保证sum>0，否则A不是正定矩阵
                // Diagonal element: must be positive, else matrix is not positive-definite
                if (sum <= 0.0) return false;
                L[i][j] = std::sqrt(sum);
            } else {
                // 下三角元素，利用已知的L[j][j]
                // Off-diagonal element: use previously computed L[j][j]
                L[i][j] = sum / L[j][j];
            }
            // 上三角部分保持为0 / Upper triangle remains zero
        }
    }
    return true;
}

/**
 * @brief 使用Cholesky下三角矩阵L将独立随机向量转换为相关随机向量  👉 用矩阵𝐿把“互不相关”的随机数变成“有关系”的随机数
 *        Use Cholesky lower-triangular matrix L to transform independent random vector into correlated vector
 *
 * @param L Cholesky下三角矩阵 / Cholesky lower-triangular matrix (K x K)
 * @param independent_vector 独立随机向量 / Independent random vector (length K)
 * @return 相关随机向量 / Correlated random vector (length K)
 *
 * 算法说明：
 * 相关向量的第i个分量为：correlated[i] = sum_{j=0}^{i} L[i][j] * independent_vector[j]
 * 即对每一行L，做下三角乘法，得到相关性。
 * 常用于将标准正态分布向量转换为具有目标相关结构的向量。
 */
std::vector<double> generate_correlated_vector(
    const std::vector<std::vector<double>>& L,
    const std::vector<double>& independent_vector) {
    
    int dim = L.size();
    // 初始化相关向量为0 / Initialize correlated vector to zero
    std::vector<double> correlated(dim, 0.0);

    // 对每个分量，做下三角乘法
    // For each component, perform lower-triangular multiplication
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j <= i; ++j) {
            correlated[i] += L[i][j] * independent_vector[j];
        }
    }
    return correlated;
}