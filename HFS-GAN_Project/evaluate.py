# evaluate.py
# ==============================================================================
#  多资产模型评估脚本
#  功能：加载训练好的生成器，并进行全面的定性、定量和金融应用评估。
#  Multivariate model evaluation script
#  Function: Load trained generator and perform comprehensive qualitative, quantitative, and financial application evaluation.
# ==============================================================================

# --- 0. 环境设置 / Environment setup ---
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import os
import sys
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf
import seaborn as sns

# 导入自定义模块 / Import custom modules
import config as cfg
from model import Generator
from data_loader import MultivariateRealFinancialDataset as RealFinancialDataset
from utils import log_and_print, setup_directories

def main():
    # --- 1. 设置与模型加载 / Setup and model loading ---
    setup_directories()  # 创建必要的目录 / Create necessary directories
    report_file = os.path.join(cfg.REPORT_DIR, "quantitative_analysis_multivariate.txt")  # 评估报告文件路径 / Path for evaluation report
    with open(report_file, "w") as f:
        f.write("--- HFS-GAN Multivariate Model Evaluation Report ---\n\n")

    log_and_print(f"--- Starting Multivariate Model Evaluation ---", report_file)
    log_and_print(f"Loading model from: {cfg.EVAL_MODEL_PATH}", report_file)
    
    generator = Generator().to(cfg.DEVICE)  # 实例化生成器并移动到设备 / Instantiate generator and move to device
    try:
        generator.load_state_dict(torch.load(cfg.EVAL_MODEL_PATH, map_location=cfg.DEVICE))  # 加载模型权重 / Load model weights
    except Exception as e:
        log_and_print(f"Error loading model: {e}", report_file)
        sys.exit()
    generator.eval()  # 设置为评估模式 / Set to evaluation mode

    # --- 2. 准备评估数据 / Prepare evaluation data ---
    log_and_print("\n--- Preparing Evaluation Data ---", report_file)
    eval_dataset = RealFinancialDataset(assets=cfg.ASSETS, start_date=cfg.START_DATE, end_date=cfg.END_DATE, seq_len=cfg.SEQ_LEN)  # 构建评估数据集 / Build evaluation dataset
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)  # 构建数据加载器 / Build data loader

    # --- 3. 生成合成数据集 / Generate synthetic dataset ---
    log_and_print(f"\n--- Generating {len(eval_dataset)} Synthetic Samples for Evaluation ---", report_file)
    fake_paths_list = []  # 存储生成路径 / Store generated paths
    real_paths_list = []  # 存储真实路径 / Store real paths
    with torch.no_grad():  # 禁用梯度计算 / Disable gradient calculation
        for real_paths_batch, real_conditions_batch in eval_dataloader:
            real_paths_list.append(real_paths_batch.numpy())  # 收集真实路径 / Collect real paths
            noise = torch.randn(real_paths_batch.size(0), cfg.NOISE_DIM, device=cfg.DEVICE)  # 生成噪声 / Generate noise
            
            # 正确解包生成器返回的元组 / Correctly unpack generator output tuple
            fake_paths_batch, _ = generator(noise, real_conditions_batch.to(cfg.DEVICE))
            
            fake_paths_list.append(fake_paths_batch.cpu().numpy())  # 收集生成路径 / Collect generated paths

    real_paths_full = np.concatenate(real_paths_list, axis=0)  # 合并所有真实路径 / Concatenate all real paths
    fake_paths_full = np.concatenate(fake_paths_list, axis=0)  # 合并所有生成路径 / Concatenate all generated paths
    log_and_print(f"Evaluation dataset ready! (Real: {real_paths_full.shape}, Generated: {fake_paths_full.shape})", report_file)

    # 计算真实和生成路径的对数收益率 / Calculate log returns for real and generated paths
    real_returns = np.log(real_paths_full[:, 1:, :] / (real_paths_full[:, :-1, :] + 1e-9))
    fake_returns = np.log(fake_paths_full[:, 1:, :] / (fake_paths_full[:, :-1, :] + 1e-9))

    # --- 4. 执行模型评估 / Perform model evaluation ---
    log_and_print("\n" + "="*50, report_file)
    log_and_print("      STARTING MODEL EVALUATION", report_file)
    log_and_print("="*50, report_file)

    # --- 4.1 定性分析 (Qualitative Analysis) ---
    log_and_print("\n--- 4.1 Qualitative Analysis: Path Gallery & Distributions ---", report_file)
    
    # 路径画廊 / Path gallery
    fig, axes = plt.subplots(3, cfg.NUM_ASSETS, figsize=(5 * cfg.NUM_ASSETS, 3 * 3), squeeze=False)
    for i in range(3):
        idx = np.random.randint(0, len(real_paths_full))  # 随机选择样本索引 / Randomly select sample index
        for j in range(cfg.NUM_ASSETS):
            ax = axes[i, j]
            ax.plot(real_paths_full[idx, :, j], color='blue', label='Real')  # 绘制真实路径 / Plot real path
            ax.plot(fake_paths_full[idx, :, j], color='red', label='Generated', alpha=0.8)  # 绘制生成路径 / Plot generated path
            ax.grid(True, linestyle='--', alpha=0.6)
            if i == 0: ax.set_title(cfg.ASSETS[j])  # 设置标题 / Set title
            if j == 0: ax.set_ylabel(f"Sample #{idx}")  # 设置y轴标签 / Set y-label
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.FIGURE_DIR, "evaluation_path_gallery.png"))  # 保存图片 / Save figure
    plt.close()

    # 回报率分布 / Return distributions
    fig, axes = plt.subplots(1, cfg.NUM_ASSETS, figsize=(6 * cfg.NUM_ASSETS, 5))
    if cfg.NUM_ASSETS == 1: axes = [axes]
    for i in range(cfg.NUM_ASSETS):
        sns.histplot(real_returns[:, :, i].flatten(), ax=axes[i], color='blue', label='Real', stat='density', bins=150)  # 真实分布 / Real distribution
        sns.histplot(fake_returns[:, :, i].flatten(), ax=axes[i], color='red', label='Generated', stat='density', bins=150)  # 生成分布 / Generated distribution
        axes[i].set_title(f"Return Distribution for {cfg.ASSETS[i]}")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.FIGURE_DIR, "evaluation_return_distributions.png"))  # 保存图片 / Save figure
    plt.close()

    # --- 4.2 定量分析 (Quantitative Analysis) ---
    log_and_print("\n--- 4.2 Quantitative Analysis: Statistical Properties ---", report_file)
    for i in range(cfg.NUM_ASSETS):
        log_and_print(f"\n--- Asset: {cfg.ASSETS[i]} ---", report_file)
        real_ret_asset = real_returns[:, :, i].flatten()  # 真实资产收益率 / Real asset returns
        fake_ret_asset = fake_returns[:, :, i].flatten()  # 生成资产收益率 / Generated asset returns
        log_and_print(f"Mean (Real vs Gen):      {np.mean(real_ret_asset):.6f} vs {np.mean(fake_ret_asset):.6f}", report_file)
        log_and_print(f"Std Dev (Real vs Gen):   {np.std(real_ret_asset):.6f} vs {np.std(fake_ret_asset):.6f}", report_file)
        log_and_print(f"Skewness (Real vs Gen):  {skew(real_ret_asset):.4f} vs {skew(fake_ret_asset):.4f}", report_file)
        log_and_print(f"Kurtosis (Real vs Gen):  {kurtosis(real_ret_asset, fisher=False):.4f} vs {kurtosis(fake_ret_asset, fisher=False):.4f}", report_file)

    # --- 4.3 相关性分析 (Correlation Analysis) ---
    log_and_print("\n--- 4.3 Correlation Analysis ---", report_file)
    real_returns_reshaped = np.reshape(real_returns, (-1, cfg.NUM_ASSETS))  # 真实收益率重塑 / Reshape real returns
    fake_returns_reshaped = np.reshape(fake_returns, (-1, cfg.NUM_ASSETS))  # 生成收益率重塑 / Reshape generated returns
    real_corr_df = pd.DataFrame(real_returns_reshaped, columns=cfg.ASSETS).corr()  # 真实相关矩阵 / Real correlation matrix
    fake_corr_df = pd.DataFrame(fake_returns_reshaped, columns=cfg.ASSETS).corr()  # 生成相关矩阵 / Generated correlation matrix

    log_and_print("Real Returns Correlation Matrix:", report_file)
    log_and_print(real_corr_df, report_file)
    log_and_print("\nGenerated Returns Correlation Matrix:", report_file)
    log_and_print(fake_corr_df, report_file)
    
    corr_diff = np.linalg.norm(real_corr_df.values - fake_corr_df.values, 'fro')  # 相关矩阵差的Frobenius范数 / Frobenius norm of correlation matrix difference
    log_and_print(f"\nFrobenius norm of correlation matrix difference: {corr_diff:.4f}", report_file)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(real_corr_df, ax=axes[0], annot=True, cmap='coolwarm', vmin=-1, vmax=1)  # 真实相关热力图 / Real correlation heatmap
    axes[0].set_title('Real Data Correlation')
    sns.heatmap(fake_corr_df, ax=axes[1], annot=True, cmap='coolwarm', vmin=-1, vmax=1)  # 生成相关热力图 / Generated correlation heatmap
    axes[1].set_title('Generated Data Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.FIGURE_DIR, "evaluation_correlation_matrix.png"))  # 保存图片 / Save figure
    plt.close()

    # --- 4.4 金融应用评估 (Financial Application) ---
    log_and_print("\n--- 4.4 Financial Application: Portfolio Risk Metrics ---", report_file)
    weights = np.array([1/cfg.NUM_ASSETS] * cfg.NUM_ASSETS)  # 等权重投资组合 / Equal-weighted portfolio
    
    # 我们关心的是整个序列（63天）结束后的总回报率 / Focus on total return at end of sequence (e.g. 63 days)
    real_portfolio_total_returns = (real_paths_full[:, -1, :] @ weights) - 1.0
    fake_portfolio_total_returns = (fake_paths_full[:, -1, :] @ weights) - 1.0

    def calculate_risk_metrics(portfolio_returns, confidence_level=0.95):
        # 计算VaR和ES风险指标 / Calculate VaR and ES risk metrics
        var_quantile = 1.0 - confidence_level
        VaR = np.quantile(portfolio_returns, var_quantile)  # VaR: 分位数 / Value at Risk: quantile
        ES = portfolio_returns[portfolio_returns < VaR].mean()  # ES: VaR以下的均值 / Expected Shortfall: mean below VaR
        return VaR, ES

    real_VaR, real_ES = calculate_risk_metrics(real_portfolio_total_returns)
    fake_VaR, fake_ES = calculate_risk_metrics(fake_portfolio_total_returns)

    log_and_print(f"Portfolio: Equal-weighted {cfg.ASSETS}", report_file)
    log_and_print(f"Real Data Portfolio VaR (95%): {real_VaR:.4%}", report_file)
    log_and_print(f"Generated Data Portfolio VaR (95%): {fake_VaR:.4%}", report_file)
    log_and_print("-" * 30, report_file)
    log_and_print(f"Real Data Portfolio ES (95%): {real_ES:.4%}", report_file)
    log_and_print(f"Generated Data Portfolio ES (95%): {fake_ES:.4%}", report_file)

    log_and_print("\n" + "="*50, report_file)
    log_and_print("      MODEL EVALUATION FINISHED", report_file)
    log_and_print("="*50, report_file)

if __name__ == "__main__":
    main()  # 程序入口 / Entry point