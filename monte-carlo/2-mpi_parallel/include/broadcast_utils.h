#ifndef BROADCAST_UTILS_H
#define BROADCAST_UTILS_H

#include <vector>
#include "debtor.h"

void broadcast_portfolio(std::vector<Debtor>& portfolio, int root_rank);
void broadcast_matrix(std::vector<std::vector<double>>& matrix, int root_rank);

#endif // BROADCAST_UTILS_H