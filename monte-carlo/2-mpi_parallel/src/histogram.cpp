// histogram.cpp
#include "histogram.h"
#include <numeric>
#include <stdexcept>
#include <iostream>

Histogram::Histogram(double min_val, double max_val, int num_bins)
    : min_val(min_val),
      max_val(max_val),
      num_bins(num_bins),
      bin_width((max_val - min_val) / num_bins),
      bins(num_bins, 0) {}


// Adds a value to the histogram, incrementing the appropriate bin
void Histogram::add(double value) {
    if (value < min_val) {
        bins[0]++;
    } else if (value >= max_val) {
        bins[num_bins - 1]++;
    } else {
        int bin_index = static_cast<int>((value - min_val) / bin_width);
        bins[bin_index]++;
    }
}

// Resets the histogram data
void Histogram::set_data(const std::vector<int>& values) {
    if (values.size() != bins.size()) {
        throw std::invalid_argument("Histogram::set_data - size mismatch");
    }
    bins = values;
}

std::vector<int>& Histogram::data() {
    return bins;
}

// Returns the bin width
double Histogram::estimate_var(double alpha) const {
    int total = std::accumulate(bins.begin(), bins.end(), 0);
    int cutoff = static_cast<int>(total * alpha);

    int cumulative = 0;
    for (int i = 0; i < num_bins; ++i) {
        cumulative += bins[i];
        if (cumulative >= cutoff) {
            return min_val + i * bin_width;
        }
    }

    return max_val;
}

// Estimates the expected shortfall (ES) at a given alpha level
double Histogram::estimate_es(double alpha) const {
    int total = std::accumulate(bins.begin(), bins.end(), 0);
    int cutoff = static_cast<int>(total * alpha);

    double sum = 0.0;
    int count = 0;
    int cumulative = 0;

    for (int i = 0; i < num_bins; ++i) {
        int bin_count = bins[i];
        cumulative += bin_count;
        if (cumulative >= cutoff) {
            int over = cumulative - cutoff;
            int needed = bin_count - over;

            double bin_center = min_val + (i + 0.5) * bin_width;
            sum += bin_center * needed;
            count += needed;

            // Add remaining bins
            for (int j = i + 1; j < num_bins; ++j) {
                double center = min_val + (j + 0.5) * bin_width;
                sum += center * bins[j];
                count += bins[j];
            }
            break;
        }
    }

    if (count == 0) return max_val;
    return sum / count;
}