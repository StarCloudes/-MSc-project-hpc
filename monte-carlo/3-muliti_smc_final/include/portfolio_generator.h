#ifndef PORTFOLIO_GENERATOR_H
#define PORTFOLIO_GENERATOR_H

#include "debtor.h"
#include <vector>

inline const int NUM_FACTORS = 3;

/// @brief Generate a synthetic credit portfolio with randomized parameters.
/// @param N Number of borrowers
/// @return A vector of Debtor objects
std::vector<Debtor> generate_synthetic_portfolio(int N);

#endif // PORTFOLIO_GENERATOR_H