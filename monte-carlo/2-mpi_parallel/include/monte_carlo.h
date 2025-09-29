#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include "debtor.h"
#include <vector>
#include <random>

/// @brief Run Monte Carlo simulation for credit risk.
/// @param portfolio List of debtors
/// @param L Cholesky lower-triangular matrix of correlation
/// @param num_paths Number of Monte Carlo iterations
/// @param report_interval Interval for printing progress to console
/// @param max_seconds Maximum allowed execution time in seconds
/// @return Vector of total loss in each simulation path
std::vector<double> simulate_monte_carlo(const std::vector<Debtor>& portfolio,
                                         const std::vector<std::vector<double>>& L,
                                         int num_paths,
                                         int report_interval,
                                         double max_seconds,
                                         std::mt19937& gen,
                                         int mpi_rank);
#endif // MONTE_CARLO_H