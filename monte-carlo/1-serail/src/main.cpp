#include "portfolio_generator.h"
#include "cholesky.h"
#include "copula_transform.h"
#include "monte_carlo.h"
#include <iostream>
#include <numeric>    // for std::accumulate
#include <algorithm>  // for std::sort
#include <chrono>     // for timing
#include <vector>
#include <random>
#include "debtor.h" 


void test_portfolio_generation(int N) {
    auto portfolio = generate_synthetic_portfolio(N);

    std::cout << "✅ Portfolio generated with " << portfolio.size() << " debtors.\n";

    for (int i = 0; i < 5 && i < portfolio.size(); ++i) {
        const auto& d = portfolio[i];
        std::cout << "Debtor " << i
                  << " | PD=" << d.pd
                  << " | LGD=" << d.lgd
                  << " | Exposure=" << d.exposure
                  << " | Rho=" << d.rho << "\n";
    }
}

void test_cholesky_and_sampling() {
    std::vector<std::vector<double>> corr = {
        {1.0, 0.2, 0.1},
        {0.2, 1.0, 0.3},
        {0.1, 0.3, 1.0}
    };

    std::vector<std::vector<double>> L;
    if (cholesky_decompose(corr, L)) {
        auto sample = generate_correlated_gaussian(L, 3);
        std::cout << "Sample: ";
        for (auto x : sample) std::cout << x << " ";
        std::cout << "\n";
    } else {
        std::cerr << "Cholesky decomposition failed.\n";
    }
}

void test_adjusted_pd() {
    double base_pd = 0.05;
    double rho = 0.25;
    double z = -0.8; // current economic scenario factor

    double scenario_pd = adjusted_pd_gaussian_copula(base_pd, rho, z);
    std::cout << "Adjusted PD: " << scenario_pd << "\n";
}

void run_toy_model_sanity_check() {
    std::cout << "\n=== 🧪 Running Toy Model Sanity Check ===\n";

    // 1. Construct 3-debtor toy portfolio
    std::vector<Debtor> toy_portfolio = {
        {0.05, 1.0, 1000000, 0.10},  // PD, LGD, Exposure, Rho
        {0.20, 1.0, 500000, 0.20},
        {0.01, 1.0, 2000000, 0.05}
    };

    // 2. Define correlation matrix (symmetric, positive-definite)
    std::vector<std::vector<double>> corr = {
        {1.0, 0.1, 0.05},
        {0.1, 1.0, 0.15},
        {0.05, 0.15, 1.0}
    };

    std::vector<std::vector<double>> L;
    if (!cholesky_decompose(corr, L)) {
        std::cerr << "Cholesky decomposition failed.\n";
        return;
    }

    // 3. Use fixed seed for reproducibility
    std::mt19937 gen(42);
    std::normal_distribution<> norm(0.0, 1.0);
    std::uniform_real_distribution<> uniform(0.0, 1.0);

    int num_paths = 5;
    for (int i = 0; i < num_paths; ++i) {
        // Generate raw standard normals
        std::vector<double> z_raw(3);
        for (int j = 0; j < 3; ++j) z_raw[j] = norm(gen);

        // Apply Cholesky to get correlated Z
        std::vector<double> z(3, 0.0);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c <= r; ++c)
                z[r] += L[r][c] * z_raw[c];

        // Adjusted PD + simulate losses
        double loss = 0.0;
        std::vector<double> adjusted_pds;
        std::vector<int> defaults;

        for (int j = 0; j < 3; ++j) {
            double adj_pd = adjusted_pd_gaussian_copula(
                toy_portfolio[j].pd,
                toy_portfolio[j].rho,
                z[j]);

            adjusted_pds.push_back(adj_pd);

            double u = uniform(gen);
            if (u < adj_pd) {
                loss += toy_portfolio[j].lgd * toy_portfolio[j].exposure;
                defaults.push_back(1);
            } else {
                defaults.push_back(0);
            }
        }

        std::cout << "Path " << i + 1 << " | Z = [";
        for (auto zz : z) std::cout << zz << " ";
        std::cout << "] | Adjusted PD = [";
        for (auto p : adjusted_pds) std::cout << p << " ";
        std::cout << "] | Defaults = [";
        for (auto d : defaults) std::cout << d << " ";
        std::cout << "] | Loss = " << loss << "\n";
    }

    std::cout << "=== ✅ Toy Model Test Completed ===\n";
}

// Run convergence test over increasing number of simulation paths
void run_convergence_test() {
    int N = 1000;  // Number of debtors
    std::vector<int> path_counts = {100, 500, 1000, 2000, 5000, 10000};

    std::cout << "\n=== 🧪 Monte Carlo Convergence Test ===\n";

    // Step 1: Generate synthetic portfolio
    auto portfolio = generate_synthetic_portfolio(N);

    // Step 2: Create correlation matrix and do Cholesky
    std::vector<std::vector<double>> corr(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        corr[i][i] = 1.0;
        for (int j = 0; j < i; ++j) {
            double rho = std::min(portfolio[i].rho, portfolio[j].rho);
            corr[i][j] = corr[j][i] = rho;
        }
    }

    std::vector<std::vector<double>> L;
    if (!cholesky_decompose(corr, L)) {
        std::cerr << "Cholesky decomposition failed.\n";
        return;
    }

    // Step 3: Run simulations for different path counts
    for (int M : path_counts) {
        std::cout << "\n▶ Simulating with " << M << " paths...\n";

        auto start = std::chrono::high_resolution_clock::now();
        auto losses = simulate_monte_carlo(portfolio, L, M);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::sort(losses.begin(), losses.end());
        double var95 = losses[int(M * 0.95)];
        double es95 = std::accumulate(losses.begin() + int(M * 0.95), losses.end(), 0.0) / (M * 0.05);

        std::cout << "  [Result] VaR(95%): " << var95
                  << " | ES(95%): " << es95
                  << " | Time: " << elapsed.count() << "s\n";
    }

    std::cout << "=== ✅ Convergence test completed ===\n";
}

int main() {
    int N = 10000;      // Number of debtors
    int M = 100000;      // Number of Monte Carlo simulation paths

    //test_portfolio_generation(N);
    //test_cholesky_and_sampling();
    //test_adjusted_pd();
    //run_toy_model_sanity_check();
    //run_convergence_test();
    
    int report_every = 1000;
    double max_seconds = 6000.0;

    auto start_time = std::chrono::high_resolution_clock::now();
    // Step 1: Generate synthetic debtor portfolio
    std::cout << "Step 1: Generating synthetic portfolio with " << N << " debtors...\n";
    auto portfolio = generate_synthetic_portfolio(N);

    // Step 2: Construct correlation matrix using min(rho_i, rho_j)
    std::cout << "Step 2: Building correlation matrix...\n";
    std::vector<std::vector<double>> corr(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        corr[i][i] = 1.0;
        for (int j = 0; j < i; ++j) {
            double rho = std::min(portfolio[i].rho, portfolio[j].rho);
            corr[i][j] = corr[j][i] = rho;
        }
    }

    // Step 3: Perform Cholesky decomposition
    std::cout << "Step 3: Performing Cholesky decomposition...\n";
    std::vector<std::vector<double>> L;
    if (!cholesky_decompose(corr, L)) {
        std::cerr << "Cholesky decomposition failed. Exiting.\n";
        return 1;
    }

    // Step 4: Run Monte Carlo simulation
    std::cout << "Step 4: Running Monte Carlo simulation with " << M << " paths...\n";
    auto start = std::chrono::high_resolution_clock::now();

    auto losses = simulate_monte_carlo(portfolio, L, M, report_every, max_seconds);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulation time: " << elapsed.count() << " seconds\n";

    // Step 5: Sort losses and compute VaR / Expected Shortfall
    std::cout << "Step 5: Sorting losses and computing risk metrics...\n";
    std::sort(losses.begin(), losses.end());

    int valid_paths = losses.size();
    if (valid_paths < static_cast<int>(M * 0.95)) {
        std::cerr << "⚠️ Warning: Simulation stopped early, may affect VaR accuracy.\n";
    }

    size_t k = static_cast<size_t>(std::ceil(0.95 * (valid_paths + 1))) - 1;
    if (k >= valid_paths) k = valid_paths - 1; // clamp
    double var95 = losses[k];
    double tail_sum = std::accumulate(losses.begin() + k, losses.end(), 0.0);
    double es95 = tail_sum / static_cast<double>(valid_paths - k);

    auto end_time = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "✅ Monte Carlo simulation completed.\n";
    std::cout << "Portfolio size        : " << N << "\n";
    std::cout << "Simulated paths       : " << valid_paths << "\n";
    std::cout << "Total execution time  : " << runtime << "\n";
    std::cout << "Value at Risk (95%)   : " << var95 << "\n";
    std::cout << "Expected Shortfall    : " << es95 << "\n";

    return 0;
}