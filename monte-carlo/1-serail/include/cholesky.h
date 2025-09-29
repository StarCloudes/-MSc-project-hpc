#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <vector>

/// @brief Performs Cholesky decomposition on a correlation matrix.
/// @param corrMatrix Input symmetric, positive-definite correlation matrix
/// @param L Lower-triangular output matrix
/// @return true if decomposition succeeded, false otherwise
bool cholesky_decompose(const std::vector<std::vector<double>>& corrMatrix,
                        std::vector<std::vector<double>>& L);

/// @brief Generates a multivariate normal sample with correlation structure L.
/// @param L Lower-triangular matrix from Cholesky
/// @param dim Dimension of the sample
/// @return Correlated Gaussian sample vector
std::vector<double> generate_correlated_gaussian(const std::vector<std::vector<double>>& L, int dim);

#endif // CHOLESKY_H