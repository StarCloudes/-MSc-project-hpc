// gan_preprocess.h
#pragma once
#include <vector>
#include <string>
#include <random>  

namespace mcutil {

    // 类型别名
    using Mat = std::vector < std::vector < double >> ;
    using Vec = std::vector < double > ;

    // ---------- I/O & 形状 ----------
    std::vector < std::vector < double >> read_matrix_csv(const std::string & filename, int expected_cols);
    std::vector < double > flatten_row_major(const std::vector < std::vector < double >> & a);
    void partition_rows(int world_size, int M_total,
        std::vector < int > & counts_rows,
        std::vector < int > & displs_rows);

    // ---------- 统计/线代工具 ----------
    void standardize_columns_inplace(std::vector < double > & flat, int rows, int cols);
    void colorize_rows_inplace(std::vector < double > & flat, int rows, int cols,
        const Mat & L);
    std::vector < std::vector < double >> sample_corr_from_flat(const std::vector < double > & flat, int rows, int K);
    bool cholesky_with_jitter(std::vector < std::vector < double >> M, std::vector < std::vector < double >> & L,
        double jitter = 1e-10, int max_tries = 5);
    std::vector < double > forward_solve(const Mat & L,
        const std::vector < double > & x); // 解 L y = x
    void whiten_then_colorize_inplace(std::vector < double > & flat, int M, int K,
        const Mat & L_target);

    // ---------- GAN 后处理 ----------
    void quantile_map_inplace(std::vector < double > & gan_flat, int M, int K,
        const std::vector < double > & real_flat, int R);
    std::vector < double > generate_correlated_normals(int M, int K,
        const Mat & L);
    void iman_conover_reorder_inplace(std::vector < double > & flat, int M, int K,
        const Mat & L_target);

    // ---------- 采样器（对照用） ----------
    std::vector < double > generate_correlated_student_t(const Mat & L, double nu, std::mt19937 & gen);

    // ---------- 诊断 ----------
    void print_factor_diag_from_flat(const std::vector < double > & flat, int rows, int K,
        int sample_rows,
        const char * title);

} // namespace mcutil