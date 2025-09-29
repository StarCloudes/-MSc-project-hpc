#include "copula_transform.h"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <boost/math/distributions/normal.hpp> // Use Boost's normal distribution

// Standard normal distribution functions
static const boost::math::normal_distribution<> standard_normal(0.0, 1.0);

// Adjusted PD using Gaussian copula transformation
double adjusted_pd_gaussian_copula(double original_pd, double rho, double z_factor) {
    if (original_pd <= 0.0)
        return 0.0;
    if (original_pd >= 1.0)
        return 1.0;

    double inv_cdf = boost::math::quantile(standard_normal, original_pd);
    double denominator = std::sqrt(1.0 - rho);

    double transformed = (inv_cdf - std::sqrt(rho) * z_factor) / denominator;

    return boost::math::cdf(standard_normal, transformed);
}