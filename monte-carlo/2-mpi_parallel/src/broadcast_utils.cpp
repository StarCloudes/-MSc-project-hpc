#include "broadcast_utils.h"
#include <mpi.h>

void broadcast_portfolio(std::vector<Debtor>& portfolio, int root_rank) {
    int size = portfolio.size();
    MPI_Bcast(&size, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    if (portfolio.empty()) portfolio.resize(size);
    MPI_Bcast(portfolio.data(), size * sizeof(Debtor), MPI_BYTE, root_rank, MPI_COMM_WORLD);
}

void broadcast_matrix(std::vector<std::vector<double>>& matrix, int root_rank) {
    int N = matrix.size();
    if (matrix.empty()) {
        matrix.resize(N, std::vector<double>(N));
    }
    for (int i = 0; i < N; ++i) {
        MPI_Bcast(matrix[i].data(), N, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
    }
}