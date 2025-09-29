// main_mpi_reduce.cpp (Fixed + Annotated version)

#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "debtor.h"
#include "portfolio_generator.h"
#include "cholesky.h"
#include "copula_transform.h"
#include "simulator.h"
#include "monte_carlo.h"
#include "histogram.h"
#include "broadcast_utils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    double program_start_time = MPI_Wtime(); // Record start time for the entire program

    const int N = 100;         // Number of debtors
    const int M_total = 10000;  // Total number of Monte Carlo paths
    const int M_local = M_total / world_size;
    const int report_interval = 1000;

    std::mt19937 gen(world_rank + 1234); // Each rank gets unique seed

    // -----------------------
    // Step 1: Rank 0 generates data
    // -----------------------
    std::vector<Debtor> portfolio;
    std::vector<std::vector<double>> L;

    if (world_rank == 0) {
        std::cout << "[Rank 0] Generating portfolio and correlation matrix...\n";
        portfolio = generate_synthetic_portfolio(N);

        // Construct correlation matrix
        std::vector<std::vector<double>> corr(N, std::vector<double>(N, 0.0));
        for (int i = 0; i < N; ++i) {
            corr[i][i] = 1.0;
            for (int j = 0; j < i; ++j) {
                double rho = std::min(portfolio[i].rho, portfolio[j].rho);
                corr[i][j] = corr[j][i] = rho;
            }
        }

        if (!cholesky_decompose(corr, L)) {
            std::cerr << "❌ Cholesky decomposition failed\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // -----------------------
    // Step 2: Broadcast to all ranks
    // -----------------------
    if (world_rank != 0) {
        portfolio.resize(N);                           // ⬅️ allocate space for recv
        L.resize(N, std::vector<double>(N));           // ⬅️ same for matrix
    }

    if (world_rank == 0) std::cout << "[Rank 0] Broadcasting portfolio and matrix...\n";
    broadcast_portfolio(portfolio, 0);
    broadcast_matrix(L, 0);

    // -----------------------
    // Step 3: Local Simulation + Histogram
    // -----------------------
    const double loss_min = 0.0;
    const double loss_max = 1e8;
    const int num_bins = 1000;
    Histogram local_hist(loss_min, loss_max, num_bins);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < M_local; ++i) {
        std::vector<double> z = generate_stratified_correlated_gaussian(
            L, N, i + world_rank * M_local, M_total, gen);
        double loss = simulate_single_path(portfolio, z);
        local_hist.add(loss);

        if ((i + 1) % report_interval == 0) {
            double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << "[Rank " << world_rank << "] [Progress] Simulated "
                      << (i + 1) << " / " << M_local
                      << " paths (" << seconds << "s elapsed)\n";
        }
    }

    // -----------------------
    // Step 4: Allreduce Histogram bins
    // -----------------------
    std::vector<int> global_bins(num_bins);
    MPI_Allreduce(local_hist.data().data(),
                  global_bins.data(),
                  num_bins,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    double program_end_time = MPI_Wtime();  // Record end time for the entire program            

    // -----------------------
    // Step 5: Rank 0 estimates VaR and ES
    // -----------------------
    if (world_rank == 0) {
        Histogram final_hist(loss_min, loss_max, num_bins);
        final_hist.set_data(global_bins);

        double var95 = final_hist.estimate_var(0.95);
        double es95  = final_hist.estimate_es(0.95);

        std::cout << "\n✅ Monte Carlo simulation completed (MPI-Reduce).\n";
        std::cout << "Total paths     : " << M_total << "\n";
        std::cout << "Total execution time: " << (program_end_time - program_start_time) << " seconds\n";
        std::cout << "Approx. Value at Risk (95%) : " << var95 << "\n";
        std::cout << "Approx. Expected Shortfall  : " << es95 << "\n";
    }

    MPI_Finalize();
    return 0;
}