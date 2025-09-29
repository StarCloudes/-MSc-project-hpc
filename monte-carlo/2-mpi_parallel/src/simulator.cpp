#include "simulator.h"
#include <random>
#include <cmath>
#include <algorithm> 
#include <boost/math/distributions/normal.hpp>  // Boost's normal distribution


// Inverse CDF (probit) using Boost
double inverse_standard_normal(double u) {
    static const boost::math::normal normal_dist(0.0, 1.0);  // mean 0, std 1
    return boost::math::quantile(normal_dist, u);  // equivalent to Φ⁻¹(u)
}

// Generate stratified correlated Gaussian vector of dimension 'dim'
std::vector<double> generate_stratified_correlated_gaussian(
    const std::vector<std::vector<double>>& L,
    int dim,
    int path_index,
    int total_paths,
    std::mt19937& gen) {

    std::vector<double> independent(dim);
    std::vector<double> correlated(dim, 0.0);

    std::uniform_real_distribution<double> uniform(0.0, 1.0);  // not thread_local anymore

    // Step 1: Per-path stratification (NOT per-dimension); per-dimension uses independent uniforms
    const double eps = 1e-12;
    double u_path = (path_index + uniform(gen)) / total_paths; // keep path-level stratification (if needed)
    (void)u_path; // currently unused; retained for future extensions
    for (int i = 0; i < dim; ++i) {
        double u_i = std::min(std::max(uniform(gen), eps), 1.0 - eps);
        independent[i] = inverse_standard_normal(u_i);
    }
    // Step 2: Apply Cholesky transformation to induce correlation
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j <= i; ++j) {
            correlated[i] += L[i][j] * independent[j];
        }
    }

    return correlated;
}