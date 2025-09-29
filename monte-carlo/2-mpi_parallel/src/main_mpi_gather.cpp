// main_mpi.cpp
// MPI-based version of Monte Carlo credit risk simulator
#include <mpi.h>
#include "portfolio_generator.h"
#include "cholesky.h"
#include "copula_transform.h"
#include "monte_carlo.h"
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double program_start_time = MPI_Wtime(); // Record start time for the entire program

    const int N = 100;      // Number of debtors
    const int M_total = 10000;  // Total number of Monte Carlo simulation paths

    int M_local = M_total / world_size;
    int report_interval = 1000;
    double max_seconds = 4000.0;

    std::mt19937 gen(world_rank + 1234); // Unique seed per process

    // Step 1: Master generates the portfolio and correlation matrix
    std::vector<Debtor> portfolio;
    std::vector<std::vector<double>> L;

    if (world_rank == 0) {
        std::cout << "[Rank 0] Generating synthetic portfolio...\n";
        portfolio = generate_synthetic_portfolio(N);

        std::vector<std::vector<double>> corr(N, std::vector<double>(N, 0.0));
        for (int i = 0; i < N; ++i) {
            corr[i][i] = 1.0;
            for (int j = 0; j < i; ++j) {
                double rho = std::min(portfolio[i].rho, portfolio[j].rho);
                corr[i][j] = corr[j][i] = rho;
            }
        }

        std::cout << "[Rank 0] Performing Cholesky decomposition...\n";
        if (!cholesky_decompose(corr, L)) {
            std::cerr << "Cholesky decomposition failed.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast portfolio size and data
    if (world_rank != 0) {
        portfolio.resize(N);
        L.resize(N, std::vector<double>(N));
    }
    MPI_Bcast(portfolio.data(), N * sizeof(Debtor), MPI_BYTE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < N; ++i)
        MPI_Bcast(L[i].data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 2: Local Monte Carlo simulation
    auto local_losses = simulate_monte_carlo(portfolio, L, M_local, report_interval, max_seconds, gen, world_rank);
    int local_count = local_losses.size();

    // Step 3: Gather local loss sizes to master
    std::vector<int> recv_counts(world_size);
    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs;
    std::vector<double> all_losses;
    int total_paths = 0;

    if (world_rank == 0) {
        displs.resize(world_size);
        for (int i = 0; i < world_size; ++i) {
            displs[i] = total_paths;
            total_paths += recv_counts[i];
        }
        all_losses.resize(total_paths);
    }

    // Step 4: Gather variable-sized losses
    MPI_Gatherv(local_losses.data(), local_count, MPI_DOUBLE,
                all_losses.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double program_end_time = MPI_Wtime();  // Record end time for the entire program       

    // Step 5: Risk metrics on master
    if (world_rank == 0) {
        std::cout << "[Rank 0] Sorting and calculating risk metrics...\n";
        std::sort(all_losses.begin(), all_losses.end());

        int cutoff = static_cast<int>(0.95 * all_losses.size());
        double var95 = all_losses[cutoff];
        double es95 = std::accumulate(all_losses.begin() + cutoff, all_losses.end(), 0.0) /
                      (all_losses.size() - cutoff);

        std::cout << "\n\u2705 Monte Carlo simulation completed (MPI).\n";
        std::cout << "Total paths          : " << all_losses.size() << "\n";
        std::cout << "Total execution time: " << (program_end_time - program_start_time) << " seconds\n";
        std::cout << "Value at Risk (95%)  : " << var95 << "\n";
        std::cout << "Expected Shortfall (95%)  : " << es95 << "\n";
    }

    MPI_Finalize();
    return 0;
}