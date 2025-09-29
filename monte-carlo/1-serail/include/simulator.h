#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "debtor.h"
#include <vector>

/// @brief Computes total portfolio loss in a single Monte Carlo path.
/// @param portfolio Vector of debtors
/// @param economic_factors Correlated Gaussian values from Cholesky (same size as portfolio)
/// @return Total loss in this scenario
double simulate_single_path(const std::vector<Debtor>& portfolio,
                            const std::vector<double>& economic_factors);

/// @brief Generates correlated Gaussian vector using stratified sampling
/// @param L Cholesky factor of correlation matrix
/// @param dim Number of dimensions (should match portfolio size)
/// @param path_index Current path index (0 to M-1)
/// @param total_paths Total number of Monte Carlo paths (M)
/// @return Correlated standard normal vector
std::vector<double> generate_stratified_correlated_gaussian(
    const std::vector<std::vector<double>>& L,
    int dim,
    int path_index,
    int total_paths);

#endif // SIMULATOR_H