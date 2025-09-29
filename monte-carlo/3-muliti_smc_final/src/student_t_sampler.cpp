#include "student_t_sampler.h"
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include <algorithm> // for std::min/max

/**
 * @brief 使用 Student-t Copula 生成资产价值向量
 *        Generate asset value vector using Student-t Copula
 *
 * 该函数通过多维Student-t分布生成随机数，然后通过一系列变换，
 * 最终得到一组具有Student-t尾部相关性的、服从标准正态分布的资产价值。
 * This function generates random numbers from a multivariate Student-t distribution,
 * then transforms them to obtain asset values with Student-t tail dependence,
 * following a standard normal distribution.
 *
 * @param dim 向量的维度 (例如 N+1) / Dimension of the vector (e.g., N+1)
 * @param degrees_of_freedom Student-t分布的自由度 (nu) / Degrees of freedom for Student-t distribution (nu)
 * @param gen 随机数生成器 / Random number generator
 * @return std::vector<double> 最终生成的资产价值向量 / The generated asset value vector
 */
std::vector<double> generate_student_t_asset_values(
    int dim,
    double degrees_of_freedom,
    std::mt19937& gen) {

    // 初始化所需的概率分布对象
    // Initialize required distribution objects
    static const boost::math::normal standard_normal(0.0, 1.0); // 标准正态分布 / Standard normal distribution
    boost::math::students_t_distribution<> standard_t(degrees_of_freedom); // 标准Student-t分布 / Standard Student-t distribution
    boost::math::chi_squared_distribution<> chi_sq(degrees_of_freedom);    // 卡方分布 / Chi-squared distribution

    std::normal_distribution<double> normal_dist(0.0, 1.0);   // 用于生成正态随机数 / For generating normal random numbers
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0); // 用于生成[0,1]均匀随机数 / For generating uniform random numbers in [0,1]

    // 步骤 1: 生成 dim 维独立标准正态向量 Y
    // Step 1: Generate a dim-dimensional independent standard normal vector Y
    std::vector<double> y_vec(dim);
    for (int i = 0; i < dim; ++i) {
        y_vec[i] = normal_dist(gen);
    }
    
    // 步骤 2: 生成一个独立的卡方(Chi-squared)随机变量 S
    // Step 2: Generate an independent chi-squared random variable S
    // 通过对均匀分布随机数取分位数得到
    // Obtain S by applying the quantile function to a uniform random number
    double u_chi = uniform_dist(gen);
    double s_val = boost::math::quantile(chi_sq, u_chi);
    if (s_val < 1e-9) s_val = 1e-9; // 避免除以零 / Avoid division by zero

    // 步骤 3: 构造 Student-t 向量 X = Y / sqrt(S / nu)
    // Step 3: Construct Student-t vector X = Y / sqrt(S / nu)
    double divisor = std::sqrt(s_val / degrees_of_freedom);
    std::vector<double> x_vec(dim);
    for (int i = 0; i < dim; ++i) {
        x_vec[i] = y_vec[i] / divisor;
    }

    // 步骤 4 & 5: 通过 t-CDF 变换到 [0,1] 区间, 再通过逆高斯CDF变换回正态域
    // Step 4 & 5: Transform x_vec to [0,1] via Student-t CDF, then to normal domain via inverse normal CDF
    std::vector<double> asset_values(dim);
    for (int i = 0; i < dim; ++i) {
        double u_t = boost::math::cdf(standard_t, x_vec[i]); // t分布的累积分布值 / CDF value of Student-t
        // 将u_t限制在(0,1)的小区间内，避免 quantile 函数输入0或1导致无穷大
        // Clamp u_t to (0,1) to avoid infinity from quantile function
        u_t = std::min(std::max(u_t, 1e-10), 1.0 - 1e-10); 
        asset_values[i] = boost::math::quantile(standard_normal, u_t); // 逆高斯CDF变换 / Inverse normal CDF transform
    }
    
    return asset_values;
}