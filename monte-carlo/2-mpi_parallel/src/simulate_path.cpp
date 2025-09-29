#include "simulator.h"
#include "copula_transform.h"

#include <random>
#include <stdexcept>

double simulate_single_path(const std::vector<Debtor>& portfolio,
                            const std::vector<double>& economic_factors) {
    if (portfolio.size() != economic_factors.size()) {
        throw std::runtime_error("Portfolio size and factor size mismatch.");
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