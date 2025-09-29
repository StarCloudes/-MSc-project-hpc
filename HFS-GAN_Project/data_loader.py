# data_loader.py
# ==============================================================================
# 负责多资产数据的下载、对齐、预处理和封装
# Responsible for downloading, aligning, preprocessing, and packaging multi-asset data
# ==============================================================================

import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import log_and_print
import config as cfg

class MultivariateRealFinancialDataset(Dataset):
    def __init__(self, assets, start_date, end_date, seq_len):
        # 日志：开始下载多资产数据 / Log: Start downloading data for multiple assets
        log_and_print(f"Downloading data for multiple assets: {assets}...")
        self.seq_len = seq_len  # 输入序列长度 / Input sequence length
        self.assets = assets    # 资产列表 / List of assets
        all_prices = []         # 用于存储所有资产的收盘价 / Store closing prices for all assets

        # 下载每个资产的收盘价数据 / Download closing price data for each asset
        for ticker in assets:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
            data.name = ticker  # 设置列名为资产代码 / Set column name as ticker
            all_prices.append(data)
        
        # 使用内连接对齐所有资产的交易日 / Align trading days using inner join
        df_prices = pd.concat(all_prices, axis=1, join='inner')
        log_and_print(f"Aligned data from {df_prices.index.min().date()} to {df_prices.index.max().date()}")

        prices_np = df_prices.values  # 转为numpy数组 / Convert to numpy array
        returns_np = np.log(prices_np[1:] / prices_np[:-1])  # 计算对数收益率 / Calculate log returns
        
        self.paths = []      # 存储有效路径片段 / Store valid path segments
        self.conditions = [] # 存储条件向量 / Store condition vectors
        
        # 滑动窗口提取路径片段和条件 / Sliding window to extract path segments and conditions
        for i in range(len(prices_np) - self.seq_len):
            path_segment = prices_np[i : i + self.seq_len]  # 当前窗口的价格序列 / Price sequence for current window
            if np.all(path_segment > 1e-8):  # 检查数据有效性 / Check data validity
                normalized_segment = path_segment / path_segment[0]  # 归一化 / Normalize by first value
                self.paths.append(normalized_segment)
                
                # 历史收益率片段 / Historical returns segment
                hist_returns_segment = returns_np[max(0, i-252):i] if i > 0 else np.zeros((1, cfg.NUM_ASSETS))
                hist_vols = np.std(hist_returns_segment, axis=0) * np.sqrt(252)  # 年化波动率 / Annualized volatility
                hist_means = np.mean(hist_returns_segment, axis=0) * 252         # 年化均值 / Annualized mean
                condition = np.concatenate([hist_vols, hist_means])              # 条件向量拼接 / Concatenate condition vector
                self.conditions.append(condition)
        
        # 转为Tensor格式 / Convert to Tensor format
        self.paths = torch.tensor(np.array(self.paths), dtype=torch.float32)
        self.conditions = torch.tensor(np.array(self.conditions), dtype=torch.float32)
        log_and_print(f"Data ready! {len(self.paths)} valid multivariate paths extracted.")  # 日志：数据准备完成 / Log: Data ready
        
    def __len__(self):
        # 返回样本数量 / Return number of samples
        return len(self.paths)
    
    def __getitem__(self, idx):
        # 返回指定索引的路径和条件 / Return path and condition at given index
        return self.paths[idx], self.conditions[idx]