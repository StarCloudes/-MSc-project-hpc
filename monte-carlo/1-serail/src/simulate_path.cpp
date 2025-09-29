#include "simulator.h"
#include "copula_transform.h"

#include <random>
#include <stdexcept>
// Simulates a single path of credit risk losses for a portfolio of debtors
double simulate_single_path(const std::vector<Debtor>& portfolio,
                            const std::vector<double>& economic_factors) {
    // if (portfolio.size() != economic_factors.size()) {
    //     throw std::runtime_error("Portfolio size and factor size mismatch.");
    // }
    // Support two modes:
    //  A) factors.size() == N  -> per-obligor factor (current behavior)
    //  B) factors.size() == 1  -> one-factor model (all obligors share z[0])
    if (!(economic_factors.size() == portfolio.size() || economic_factors.size() == 1)) {
        throw std::runtime_error("economic_factors must be size N (per-obligor) or 1 (one-factor).");
    }
    double total_loss = 0.0;

    std::default_random_engine gen(std::random_device{}());
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (size_t i = 0; i < portfolio.size(); ++i) {
        const Debtor& d = portfolio[i];
        double z = economic_factors[i];

        double scenario_pd = adjusted_pd_gaussian_copula(d.pd, d.rho, z);

        // Simulate default
        if (uniform(gen) < scenario_pd) {
            total_loss += d.exposure * d.lgd;
        }
    }

    return total_loss;
}