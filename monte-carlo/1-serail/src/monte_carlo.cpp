#include "monte_carlo.h"
#include "copula_transform.h"
#include "simulator.h"
#include "cholesky.h"

#include <iostream>
#include <chrono>
#include <random>

std::vector<double> simulate_monte_carlo(const std::vector<Debtor>& portfolio,
                                         const std::vector<std::vector<double>>& L,
                                         int num_paths,
                                         int report_interval,
                                         double max_seconds) {
    std::vector<double> losses;
    losses.reserve(num_paths);

    int N = portfolio.size();
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_paths; ++i) {
        // Check elapsed time
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        if (elapsed > max_seconds) {
            std::cout << "⏰ Reached time limit of " << max_seconds << "s, stopping early at path " << i << ".\n";
            break;
        }

        //std::vector<double> z = generate_correlated_gaussian(L, N);
        // Generate stratified Gaussian sample (variance reduction technique)
        std::vector<double> z = generate_stratified_correlated_gaussian(L, N, i, num_paths);

        // Step 2: Calculate loss using Copula-based conditional PD
        double loss = 0.0;
        static thread_local std::mt19937 gen(std::random_device{}());
        static thread_local std::uniform_real_distribution<double> uniform(0.0, 1.0);

        for (int j = 0; j < N; ++j) {
            double scenario_pd = adjusted_pd_gaussian_copula(portfolio[j].pd, portfolio[j].rho, z[j]);
            double u = uniform(gen);
            if (u < scenario_pd) {
                loss += portfolio[j].lgd * portfolio[j].exposure;
            }
        }

        losses.push_back(loss);

        // Progress output
        if ((i + 1) % report_interval == 0) {
            std::cout << "  [Progress] Simulated " << (i + 1) << " / " << num_paths
                      << " paths (" << elapsed << "s elapsed)\n";
        }
    }

    return losses;
}