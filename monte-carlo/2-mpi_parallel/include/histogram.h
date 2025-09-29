#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>

/**
 * @brief Histogram for approximating loss distribution.
 *        Used in Monte Carlo simulation to estimate VaR and ES.
 */
class Histogram {
public:
    /**
     * @brief Construct a histogram.
     * @param min_val Minimum value of loss range.
     * @param max_val Maximum value of loss range.
     * @param num_bins Number of bins in the histogram.
     */
    Histogram(double min_val, double max_val, int num_bins);

    /**
     * @brief Add a loss value to the histogram.
     * @param value The loss value.
     */
    void add(double value);

    /**
     * @brief Set bin values (used after MPI_Allreduce).
     * @param values Vector of aggregated bin counts.
     */
    void set_data(const std::vector<int>& values);

    /**
     * @brief Get reference to bin counts (for MPI).
     * @return Reference to vector of bin counts.
     */
    std::vector<int>& data();

    /**
     * @brief Estimate Value at Risk at quantile alpha.
     * @param alpha e.g., 0.95 for 95% VaR.
     * @return Estimated VaR.
     */
    double estimate_var(double alpha) const;

    /**
     * @brief Estimate Expected Shortfall beyond quantile alpha.
     * @param alpha e.g., 0.95 for 95% ES.
     * @return Estimated ES.
     */
    double estimate_es(double alpha) const;

private:
    double min_val;
    double max_val;
    int num_bins;
    double bin_width;
    std::vector<int> bins;
};

#endif  // HISTOGRAM_H