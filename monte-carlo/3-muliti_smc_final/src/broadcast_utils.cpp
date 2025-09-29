#include "broadcast_utils.h"
#include <mpi.h>

/**
 * @brief 广播 K x K 矩阵到所有 MPI 进程
 *        Broadcast a K x K matrix to all MPI processes
 *
 * @param matrix 需要广播的矩阵 / Matrix to broadcast
 * @param root_rank 根进程编号 / Root process rank
 */
void broadcast_matrix(std::vector<std::vector<double>>& matrix, int root_rank) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 仅根进程获取矩阵维度 K，其它进程初始化为 0
    // Only root process gets matrix size K, others initialize as 0
    int K = (rank == root_rank && !matrix.empty()) ? matrix.size() : 0;
    MPI_Bcast(&K, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

    // 非根进程分配空间
    // Non-root processes allocate space
    if (rank != root_rank) {
        matrix.assign(K, std::vector<double>(K));
    }

    // 广播每一行
    // Broadcast each row
    if (K > 0) {
        for (int i = 0; i < K; ++i) {
            MPI_Bcast(matrix[i].data(), K, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
        }
    }
}

/**
 * @brief 广播支持分段和校准参数的投资组合
 *        Broadcast the portfolio with segmentation and calibrated parameters
 *
 * @param portfolio 投资组合 / Portfolio
 * @param num_factors 因子数量 / Number of factors
 * @param root_rank 根进程编号 / Root process rank
 */
void broadcast_portfolio_multifactor(std::vector<Debtor>& portfolio, int num_factors, int root_rank) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // 广播投资组合大小
    // Broadcast portfolio size
    int N = (rank == root_rank) ? portfolio.size() : 0;
    MPI_Bcast(&N, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

    // 非根进程分配空间
    // Non-root processes allocate space
    if (rank != root_rank) {
        portfolio.resize(N);
    }

    // 广播每个债务人的成员变量
    // Broadcast each debtor's member variables
    for (int i = 0; i < N; ++i) {
        // 广播基础类型成员 / Broadcast basic type members
        MPI_Bcast(&portfolio[i].id, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        MPI_Bcast(&portfolio[i].exposure, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
        MPI_Bcast(&portfolio[i].pd, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
        MPI_Bcast(&portfolio[i].industry_id, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        MPI_Bcast(&portfolio[i].r_squared, 1, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);

        // --- 处理 enum class 类型 ---
        // --- Handle enum class type ---
        // 发送前转换为整数 / Convert to int before sending
        int rating_as_int = static_cast<int>(portfolio[i].rating);
        MPI_Bcast(&rating_as_int, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        // 接收后转换回 enum class / Convert back to enum class after receiving
        if (rank != root_rank) {
            portfolio[i].rating = static_cast<CreditRating>(rating_as_int);
        }

        // 广播动态向量成员 / Broadcast dynamic vector member
        if (rank != root_rank) {
            portfolio[i].factor_loadings.resize(num_factors);
        }
        MPI_Bcast(portfolio[i].factor_loadings.data(), num_factors, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
    }
}