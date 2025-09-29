#ifndef DEBTOR_H
#define DEBTOR_H

struct Debtor {
    double pd;        // Probability of Default
    double lgd;       // Loss Given Default
    double exposure;  // Exposure at Default
    double rho;       // Correlation with economic factor
};

#endif // DEBTOR_H