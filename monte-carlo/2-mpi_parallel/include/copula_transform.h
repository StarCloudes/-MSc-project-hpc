#ifndef COPULA_TRANSFORM_H
#define COPULA_TRANSFORM_H

/// @brief Computes adjusted PD under Gaussian Copula transformation.
/// @param original_pd Original (base) PD of the debtor
/// @param rho Correlation with economic factor
/// @param z_factor Systematic economic scenario factor (from multivariate Gaussian)
/// @return Adjusted scenario PD
double adjusted_pd_gaussian_copula(double original_pd, double rho, double z_factor);

#endif // COPULA_TRANSFORM_H