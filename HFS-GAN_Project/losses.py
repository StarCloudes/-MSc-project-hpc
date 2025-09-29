# losses.py
# ==============================================================================
#  v24 - 新增直接波动率匹配损失 (compute_volatility_loss)
# ==============================================================================

import torch

# --- 新增的、最直接的波动率匹配损失 ---
# Direct volatility matching loss
def compute_volatility_loss(real_returns, fake_returns):
    """
    直接计算并惩罚真实收益率和生成收益率在波动率上的差异。
    Directly compute and penalize the difference in volatility between real and generated returns.
    """
    # 计算每个资产的波动率 (标准差) / Compute volatility (std) for each asset
    real_vols = torch.std(real_returns, dim=1) # Shape: (batch_size, num_assets)
    fake_vols = torch.std(fake_returns, dim=1) # Shape: (batch_size, num_assets)
    
    # 计算波动率差异的L1损失，并在批次和资产维度上取平均
    # Compute L1 loss of volatility difference, averaged over batch and assets
    loss = torch.mean(torch.abs(real_vols - fake_vols))
    return loss

def _single_asset_stylized_fact_loss(real_returns_sa, fake_returns_sa):
    """计算单个资产的风格化事实损失的辅助函数
    Helper for stylized fact loss of a single asset
    """
    def autocorr(series):
        # 计算自相关 / Compute autocorrelation
        series = series - torch.mean(series, dim=1, keepdim=True)
        var = torch.var(series, dim=1, keepdim=True, unbiased=False)
        acorr = torch.mean(series[:, :-1] * series[:, 1:], dim=1) / (var.squeeze(-1) + 1e-8)
        return acorr
    # 波动聚集损失 / Volatility clustering loss
    loss_vol_cluster = torch.mean(torch.abs(autocorr(torch.abs(real_returns_sa)) - autocorr(torch.abs(fake_returns_sa))))
    
    def kurtosis(series):
        # 计算峰度 / Compute kurtosis
        mean = torch.mean(series, dim=1, keepdim=True)
        std = torch.std(series, dim=1, keepdim=True, unbiased=False)
        return torch.mean(((series - mean) / (std + 1e-8)) ** 4, dim=1)
    # 厚尾损失 / Fat tails loss
    loss_fat_tails = torch.mean(torch.abs(kurtosis(real_returns_sa) - kurtosis(fake_returns_sa)))
    
    return loss_vol_cluster + loss_fat_tails

def compute_stylized_fact_loss(real_returns, fake_returns):
    """计算多个资产的平均风格化事实损失
    Compute average stylized fact loss for multiple assets
    """
    total_loss = 0.0
    num_assets = real_returns.shape[-1]
    for i in range(num_assets):
        total_loss += _single_asset_stylized_fact_loss(real_returns[..., i], fake_returns[..., i])
    return total_loss / num_assets

def compute_correlation_loss(real_returns, fake_returns):
    """计算相关性矩阵的差异
    Compute difference between correlation matrices
    """
    batch_size = real_returns.shape[0]
    loss = 0.0
    for i in range(batch_size):
        corr_real = torch.corrcoef(real_returns[i].T)  # 真实相关矩阵 / Real correlation matrix
        corr_fake = torch.corrcoef(fake_returns[i].T)  # 生成相关矩阵 / Generated correlation matrix
        if torch.isnan(corr_real).any() or torch.isnan(corr_fake).any(): 
            continue  # 跳过包含NaN的样本 / Skip samples with NaN
        loss += torch.norm(corr_real - corr_fake, p='fro')  # Frobenius范数 / Frobenius norm
    return loss / batch_size if batch_size > 0 else 0.0

def compute_gradient_penalty(critic, real_samples, fake_samples, condition):
    """计算WGAN-GP的梯度惩罚
    Compute gradient penalty for WGAN-GP
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device).expand_as(real_samples)  # 随机插值系数 / Random interpolation coefficient
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)  # 插值样本 / Interpolated samples
    d_interpolated = critic(interpolated, condition)  # 判别器输出 / Critic output
    grad_outputs = torch.ones(d_interpolated.size(), device=real_samples.device, requires_grad=False)  # 梯度输出 / Gradient outputs
    
    # 计算梯度 / Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)  # 展平梯度 / Flatten gradients
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # GP损失 / Gradient penalty loss
    return gradient_penalty