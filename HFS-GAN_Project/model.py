# model.py
# ==============================================================================
#  v22.1 - 定义支持多资产的 Generator 和能感知相关性的 Critic
#  v22.1 - Define multi-asset Generator and correlation-aware Critic
# ==============================================================================

import torch
import torch.nn as nn
import config as cfg

# --- 辅助模块 (Helper Modules) ---
class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_gate = nn.Linear(in_features, out_features)  # 门控线性层 / Gating linear layer
        self.linear_value = nn.Linear(in_features, out_features) # 值线性层 / Value linear layer
        self.activation = nn.SiLU()                              # SiLU激活函数 / SiLU activation
    def forward(self, x):
        # SwiGLU激活：门控乘以值 / SwiGLU activation: gate * value
        return self.activation(self.linear_gate(x)) * self.linear_value(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)  # 第一层归一化 / First layer normalization
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)  # 多头自注意力 / Multi-head self-attention
        self.dropout1 = nn.Dropout(dropout) # Dropout层 / Dropout layer
        self.norm2 = nn.LayerNorm(d_model)  # 第二层归一化 / Second layer normalization
        # 前馈网络 / Feedforward network
        self.feed_forward = nn.Sequential(
            SwiGLU(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, src):
        normed_src = self.norm1(src)  # 归一化输入 / Normalize input
        src2 = self.self_attn(normed_src, normed_src, normed_src)[0]  # 自注意力输出 / Self-attention output
        src = src + self.dropout1(src2)  # 残差连接 / Residual connection
        normed_src = self.norm2(src)
        src2 = self.feed_forward(normed_src)  # 前馈输出 / Feedforward output
        src = src + self.dropout2(src2)       # 残差连接 / Residual connection
        return src

# --- 生成器 (Generator) ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 噪声和条件拼接后初始线性层 / Initial linear layer after concatenating noise and condition
        self.init_linear = nn.Linear(cfg.NOISE_DIM + cfg.COND_DIM, cfg.SEQ_LEN * cfg.D_MODEL)
        # 一维卷积特征提取器 / 1D CNN feature extractor
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(cfg.D_MODEL, cfg.D_MODEL, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2)
        )
        # 多层Transformer编码器 / Multiple Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg.D_MODEL, cfg.N_HEAD, cfg.D_MODEL * 2, cfg.DROPOUT)
            for _ in range(cfg.GEN_ENCODER_LAYERS)
        ])
        # 输出层：生成对数收益率 / Output layer: generate log returns
        self.returns_output_layer = nn.Sequential(
            nn.Linear(cfg.D_MODEL, cfg.NUM_ASSETS), # 输出资产数 / Output number of assets
            nn.Tanh()
        )
        self.return_scaler = 0.1  # 对数收益率缩放因子 / Log return scaling factor

    def forward(self, noise, condition):
        # 拼接噪声和条件向量 / Concatenate noise and condition
        z = torch.cat([noise, condition], dim=1)
        batch_size = z.size(0)
        # 初始线性变换并reshape为序列 / Initial linear transform and reshape to sequence
        x = self.init_linear(z).view(batch_size, cfg.SEQ_LEN, cfg.D_MODEL)
        # CNN特征提取 / CNN feature extraction
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn_extractor(x_cnn)
        x = x + x_cnn.permute(0, 2, 1)  # 残差融合CNN特征 / Residual fusion of CNN features
        # Transformer编码器 / Transformer encoder
        for block in self.transformer_blocks:
            x = block(x)
        # 生成对数收益率 / Generate log returns
        log_returns = self.returns_output_layer(x) * self.return_scaler
        # 初始价格设为1 / Set initial price to 1
        initial_price = torch.ones(batch_size, 1, cfg.NUM_ASSETS, device=cfg.DEVICE)
        # 通过对数收益率生成价格路径 / Generate price path from log returns
        price_path = torch.cat([initial_price, torch.exp(torch.cumsum(log_returns, dim=1))], dim=1)
        # 返回价格路径和对数收益率 / Return price path and log returns
        return price_path[:, :-1, :], log_returns

# --- 判别器 (Critic) ---
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入层：资产数到隐藏维度 / Input layer: num_assets to hidden dimension
        self.processor = nn.Linear(cfg.NUM_ASSETS, cfg.D_MODEL)
        # 条件向量嵌入层 / Condition embedding layer
        self.cond_linear = nn.Linear(cfg.COND_DIM, cfg.D_MODEL)
        # 多层Transformer编码器 / Multiple Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg.D_MODEL, cfg.N_HEAD, cfg.D_MODEL * 2, cfg.DROPOUT)
            for _ in range(cfg.CRITIC_ENCODER_LAYERS)
        ])
        # 相关性特征数量 / Number of correlation features
        num_corr_features = cfg.NUM_ASSETS * (cfg.NUM_ASSETS - 1) // 2
        # 输出层：结合时序特征和相关性特征 / Output layer: combine time series and correlation features
        self.output_mlp = nn.Linear(cfg.D_MODEL + num_corr_features, 1)

    def forward(self, price_path, condition):
        # 1. 提取时间序列特征 / Extract time series features
        x = self.processor(price_path)
        cond_embedding = self.cond_linear(condition).unsqueeze(1)  # 条件嵌入 / Condition embedding
        x = x + cond_embedding
        for block in self.transformer_blocks:
            x = block(x)
        x_features = x.mean(dim=1)  # 序列特征均值 / Mean of sequence features
        
        # 2. 提取相关性特征 (核心创新) / Extract correlation features (core innovation)
        log_returns = torch.log(price_path[:, 1:] / (price_path[:, :-1] + 1e-9))  # 计算对数收益率 / Compute log returns
        returns_T = log_returns.permute(0, 2, 1)  # 转置为(batch, asset, time) / Transpose to (batch, asset, time)
        mean_returns = torch.mean(returns_T, dim=2, keepdim=True)  # 均值 / Mean
        centered_returns = returns_T - mean_returns  # 去中心化 / Centered returns
        covariance_matrix = torch.bmm(centered_returns, centered_returns.transpose(1, 2)) / (returns_T.shape[2] - 1)  # 协方差矩阵 / Covariance matrix
        std_devs = torch.std(returns_T, dim=2, keepdim=True)  # 标准差 / Standard deviation
        std_dev_matrix = torch.bmm(std_devs, std_devs.transpose(1, 2))  # 标准差矩阵 / Std dev matrix
        correlation_matrix = covariance_matrix / (std_dev_matrix + 1e-8)  # 相关性矩阵 / Correlation matrix
        
        indices = torch.triu_indices(cfg.NUM_ASSETS, cfg.NUM_ASSETS, offset=1, device=cfg.DEVICE)  # 上三角索引 / Upper triangle indices
        corr_features = correlation_matrix[:, indices[0], indices[1]]  # 提取相关性特征 / Extract correlation features
        corr_features = torch.nan_to_num(corr_features) # 处理可能的NaN值 / Handle possible NaN values
        
        # 3. 组合所有特征并给出最终评分 / Combine all features and output final score
        combined_features = torch.cat([x_features, corr_features], dim=1)
        return self.output_mlp(combined_features)  # 输出判别分数 / Output critic score