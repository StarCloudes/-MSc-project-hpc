// histogram.cpp
// 1 初始化 (Constructor)=》想象成：你在地上摆了很多桶（每个桶是一个分箱 bin），等着往里扔石头（样本）。
// 2 添加数据 (add)=》就像你在考试分数段统计里，把 70 分扔进“70–80分”这个区间。
// 3 VaR 计算 (estimate_var)=》类比：考试成绩排个队，从低到高，第 95% 的人对应的分数 → 就是 VaR。
// 4 ES 计算 (estimate_es)=》类比：先找到班级里排在 95% 的人，然后算出他后面所有人的平均分 → 这就是 ES。

// 它的好处：不用保存所有样本，只要数一数桶里的个数，就能大概知道尾部风险。
// 它的不足：精度取决于桶的宽度；桶太大，就不够精细。
// 节省内存、并行计算方便（每个进程可以在本地建自己的直方图、最后用 MPI_Reduce 合并各进程的分箱计数就行）、我们算 VaR/ES，只需要近似分布，不需要每个样本精确值

#include "histogram.h"
#include <numeric>
#include <stdexcept>
#include <iostream>

/**
 * @brief 构造函数，初始化直方图参数和分箱
 *        Constructor: initialize histogram parameters and bins
 * @param min_val 损失区间最小值 / Minimum value of loss range
 * @param max_val 损失区间最大值 / Maximum value of loss range
 * @param num_bins 分箱数量 / Number of bins
 */
Histogram::Histogram(double min_val, double max_val, int num_bins)
    : min_val(min_val),
      max_val(max_val),
      num_bins(num_bins),
      bin_width((max_val - min_val) / num_bins), // 每个分箱的宽度 / Width of each bin
      bins(num_bins, 0) {} // 初始化分箱计数为0 / Initialize bin counts to zero

/**
 * @brief 向直方图添加一个损失值
 *        Add a loss value to the histogram
 * @param value 损失值 / Loss value
 */
void Histogram::add(double value) {
    // 如果值小于最小值，计入第一个分箱 / If value < min_val, count in first bin
    if (value < min_val) {
        bins[0]++;
    }
    // 如果值大于等于最大值，计入最后一个分箱 / If value >= max_val, count in last bin
    else if (value >= max_val) {
        bins[num_bins - 1]++;
    }
    // 否则，计算对应分箱索引并计数 / Otherwise, compute bin index and count
    else {
        int bin_index = static_cast<int>((value - min_val) / bin_width);
        bins[bin_index]++;
    }
}

/**
 * @brief 设置分箱计数（用于MPI归并后）
 *        Set bin counts (used after MPI reduction)
 * @param values 分箱计数向量 / Vector of bin counts
 */
void Histogram::set_data(const std::vector<int>& values) {
    if (values.size() != bins.size()) {
        throw std::invalid_argument("Histogram::set_data - size mismatch");
    }
    bins = values;
}

/**
 * @brief 获取分箱计数的引用（可修改）
 *        Get reference to bin counts (modifiable)
 * @return 分箱计数引用 / Reference to bin counts
 */
std::vector<int>& Histogram::data() {
    return bins;
}

/**
 * @brief 获取分箱计数的常量引用（只读）
 *        Get const reference to bin counts (read-only)
 * @return 分箱计数常量引用 / Const reference to bin counts
 */
const std::vector<int>& Histogram::data() const {
    return bins;
}

/**
 * @brief 估算指定分位点的VaR
 *        Estimate Value-at-Risk (VaR) at given quantile
 * @param alpha 分位点（如0.95表示95% VaR）/ Quantile (e.g., 0.95 for 95% VaR)
 * @return 估算的VaR / Estimated VaR
 */
double Histogram::estimate_var(double alpha) const {
    int total = std::accumulate(bins.begin(), bins.end(), 0); // 总样本数 / Total sample count
    int cutoff = static_cast<int>(total * alpha); // VaR分位点对应的累计计数 / Cumulative count for VaR

    int cumulative = 0;
    for (int i = 0; i < num_bins; ++i) {
        cumulative += bins[i];
        // 找到累计计数超过cutoff的分箱，返回该分箱的左边界作为VaR
        // Find the bin where cumulative count exceeds cutoff, return its left boundary as VaR
        if (cumulative >= cutoff) {
            return min_val + i * bin_width;
        }
    }
    // 如果没有找到，返回最大值 / If not found, return max_val
    return max_val;
}

/**
 * @brief 估算超过指定分位点的ES
 *        Estimate Expected Shortfall (ES) above given quantile
 * @param alpha 分位点（如0.95表示95% ES）/ Quantile (e.g., 0.95 for 95% ES)
 * @return 估算的ES / Estimated ES
 */
double Histogram::estimate_es(double alpha) const {
    int total = std::accumulate(bins.begin(), bins.end(), 0); // 总样本数 / Total sample count
    int cutoff = static_cast<int>(total * alpha); // ES分位点对应的累计计数 / Cumulative count for ES

    double sum = 0.0; // ES分子累加 / Numerator for ES
    int count = 0;    // ES分母累加 / Denominator for ES
    int cumulative = 0;

    for (int i = 0; i < num_bins; ++i) {
        int bin_count = bins[i];
        cumulative += bin_count;
        if (cumulative >= cutoff) {
            // 当前分箱只取部分计数 / Only take part of current bin
            int over = cumulative - cutoff;
            int needed = bin_count - over;

            double bin_center = min_val + (i + 0.5) * bin_width; // 分箱中心 / Bin center
            sum += bin_center * needed;
            count += needed;

            // 之后所有分箱全部计入ES / All subsequent bins are included in ES
            for (int j = i + 1; j < num_bins; ++j) {
                double center = min_val + (j + 0.5) * bin_width;
                sum += center * bins[j];
                count += bins[j];
            }
            break;
        }
    }

    // 防止除零错误 / Prevent division by zero
    if (count == 0) return max_val;
    return sum / count;
}