#include "portfolio_generator.h"
#include <random>

/// @brief Implementation of synthetic portfolio generator.
/// Uses uniform distributions to create randomized borrower parameters.
std::vector<Debtor> generate_synthetic_portfolio(int N) {
    std::vector<Debtor> portfolio;
    portfolio.reserve(N); // Reserve memory for efficiency

    std::default_random_engine gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> pd_dist(0.01, 0.2);
    std::uniform_real_distribution<double> lgd_dist(0.2, 0.6);
    std::uniform_real_distribution<double> exp_dist(10000.0, 100000.0);
    std::uniform_real_distribution<double> rho_dist(0.1, 0.3);

    for (int i = 0; i < N; ++i) {
        Debtor d;
        d.pd = pd_dist(gen);
        d.lgd = lgd_dist(gen);
        d.exposure = exp_dist(gen);
        d.rho = rho_dist(gen);
        portfolio.push_back(d);
    }

    return portfolio;
}