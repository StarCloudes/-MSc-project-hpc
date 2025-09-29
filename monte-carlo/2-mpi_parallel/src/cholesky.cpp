#include "cholesky.h"
#include <random>
#include <cmath>

bool cholesky_decompose(const std::vector<std::vector<double>>& A,
                        std::vector<std::vector<double>>& L) {
    int n = A.size();
    L.assign(n, std::vector<double>(n, 0.0));

    // Symmetrize and add small jitter to the diagonal for numerical stability
    std::vector<std::vector<double>> B = A;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double aij = A[i][j];
            double aji = A[j][i];
            B[i][j] = 0.5 * (aij + aji);
        }
    }
    const double jitter = 1e-10;
    for (int i = 0; i < n; ++i) B[i][i] += jitter;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = B[i][j];

            for (int k = 0; k < j; ++k)
                sum -= L[i][k] * L[j][k];

            if (i == j) {
                if (sum <= 0.0)
                    return false; // not positive definite
                L[i][j] = std::sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }

    return true;
}

// Modified to accept external random generator for parallel consistency
std::vector<double> generate_correlated_gaussian(const std::vector<std::vector<double>>& L,
                                                 int dim,
                                                 std::mt19937& gen) {
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> Z(dim), result(dim);
    for (int i = 0; i < dim; ++i)
        Z[i] = dist(gen);

    for (int i = 0; i < dim; ++i) {
        result[i] = 0.0;
        for (int j = 0; j <= i; ++j)
            result[i] += L[i][j] * Z[j];
    }

    return result;
}