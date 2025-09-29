# main.py
# ==============================================================================
#  v24 - 在总损失中加入新的波动率匹配损失 (volatility_loss)
#  v24 - Add new volatility matching loss to total loss
# ==============================================================================

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import numpy as np

# 导入自定义模块 / Import custom modules
import config as cfg
from model import Generator, Critic
from data_loader import MultivariateRealFinancialDataset as RealFinancialDataset
# --- MODIFICATION: Import the new loss function ---
from losses import compute_stylized_fact_loss, compute_correlation_loss, compute_gradient_penalty, compute_volatility_loss
from utils import log_and_print, setup_directories

def main():
    # --- 1. 设置 (Setup) ---
    setup_directories()  # 创建必要的目录 / Create necessary directories
    log_and_print("--- Starting HFS-GAN Training (v24 Final Solution) ---")  # 日志：开始训练 / Log: Start training
    log_and_print(f"Using device: {cfg.DEVICE}")  # 日志：使用的设备 / Log: Device used

    # --- 2. 初始化 (Initialization) ---
    generator = Generator().to(cfg.DEVICE)  # 初始化生成器并移动到设备 / Initialize generator and move to device
    critic = Critic().to(cfg.DEVICE)        # 初始化判别器并移动到设备 / Initialize critic and move to device
    
    optimizer_g = optim.Adam(generator.parameters(), lr=cfg.LR_G, betas=(cfg.BETA_1, cfg.BETA_2))  # 生成器优化器 / Generator optimizer
    optimizer_c = optim.Adam(critic.parameters(), lr=cfg.LR_C, betas=(cfg.BETA_1, cfg.BETA_2))     # 判别器优化器 / Critic optimizer

    dataset = RealFinancialDataset(assets=cfg.ASSETS, start_date=cfg.START_DATE, end_date=cfg.END_DATE, seq_len=cfg.SEQ_LEN)  # 构建数据集 / Build dataset
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)  # 构建数据加载器 / Build data loader

    # --- 3. 训练循环 (Training Loop) ---
    train_start_time = time.time()  # 记录训练开始时间 / Record training start time
    for epoch in range(cfg.EPOCHS):  # 遍历所有训练轮数 / Loop over all epochs
        epoch_start_time = time.time()  # 记录当前轮开始时间 / Record epoch start time
        for i, (real_paths, real_conditions) in enumerate(dataloader):  # 遍历每个批次 / Loop over each batch
            real_paths = real_paths.to(cfg.DEVICE)         # 移动真实路径到设备 / Move real paths to device
            real_conditions = real_conditions.to(cfg.DEVICE)  # 移动条件到设备 / Move conditions to device
            
            # --- 训练判别器 (Train Critic) ---
            optimizer_c.zero_grad()  # 清空判别器梯度 / Zero critic gradients
            noise = torch.randn(real_paths.size(0), cfg.NOISE_DIM, device=cfg.DEVICE)  # 生成噪声 / Generate noise
            with torch.no_grad():
                fake_paths, _ = generator(noise, real_conditions)  # 生成假路径 / Generate fake paths
            
            real_validity = critic(real_paths, real_conditions)  # 判别器对真实样本的输出 / Critic output for real samples
            fake_validity = critic(fake_paths, real_conditions)  # 判别器对生成样本的输出 / Critic output for fake samples
            gradient_penalty = compute_gradient_penalty(critic, real_paths, fake_paths, real_conditions)  # 计算梯度惩罚 / Compute gradient penalty
            loss_c = -torch.mean(real_validity) + torch.mean(fake_validity) + cfg.LAMBDA_GP * gradient_penalty  # 判别器损失 / Critic loss
            
            loss_c.backward()  # 反向传播 / Backpropagation
            optimizer_c.step() # 更新判别器参数 / Update critic parameters

            # --- 训练生成器 (Train Generator) ---
            if i % cfg.N_CRITIC == 0:  # 每N_CRITIC步训练一次生成器 / Train generator every N_CRITIC steps
                optimizer_g.zero_grad()  # 清空生成器梯度 / Zero generator gradients
                noise_g = torch.randn(real_paths.size(0), cfg.NOISE_DIM, device=cfg.DEVICE)  # 生成噪声 / Generate noise
                gen_paths, fake_log_returns = generator(noise_g, real_conditions)  # 生成路径和对数收益率 / Generate paths and log returns
                
                adversarial_loss = -torch.mean(critic(gen_paths, real_conditions))  # 对抗损失 / Adversarial loss
                
                real_log_returns = torch.log(real_paths[:, 1:, :] / (real_paths[:, :-1, :] + 1e-9))  # 计算真实对数收益率 / Compute real log returns
                
                # --- MODIFICATION: Calculate all loss components ---
                style_loss = compute_stylized_fact_loss(real_log_returns, fake_log_returns)  # 风格化事实损失 / Stylized fact loss
                correlation_loss = compute_correlation_loss(real_log_returns, fake_log_returns)  # 相关性损失 / Correlation loss
                # --- NEW: Calculate the direct volatility loss ---
                volatility_loss = compute_volatility_loss(real_log_returns, fake_log_returns)  # 波动率损失 / Volatility loss
                
                # --- MODIFICATION: Assemble the final, balanced total loss ---
                loss_g = (adversarial_loss + 
                          cfg.LAMBDA_STYLE * style_loss + 
                          cfg.LAMBDA_CORR * correlation_loss +
                          cfg.LAMBDA_VOL * volatility_loss)  # 生成器总损失 / Total generator loss
                
                loss_g.backward()  # 反向传播 / Backpropagation
                optimizer_g.step() # 更新生成器参数 / Update generator parameters

        # --- 4. 日志与保存 (Logging and Saving) ---
        epoch_end_time = time.time()  # 记录当前轮结束时间 / Record epoch end time
        log_message = (f"Epoch [{epoch+1}/{cfg.EPOCHS}] | C Loss: {loss_c.item():.4f} | G Loss: {loss_g.item():.4f} | "
                       f"Vol Loss: {volatility_loss.item():.4f} | Corr Loss: {correlation_loss.item():.4f} | "
                       f"Time: {epoch_end_time - epoch_start_time:.2f}s")  # 构建日志信息 / Build log message
        log_and_print(log_message)  # 输出日志 / Print log
        
        # --- MODIFICATION: Added full saving and visualization logic ---
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:  # 每隔SAVE_INTERVAL轮保存一次模型 / Save model every SAVE_INTERVAL epochs
            log_and_print(f"--- Saving checkpoint at epoch {epoch+1} ---")  # 日志：保存模型 / Log: Save model
            torch.save(generator.state_dict(), os.path.join(cfg.MODEL_DIR, f"generator_epoch_{epoch+1}.pth"))  # 保存生成器权重 / Save generator weights
            torch.save(critic.state_dict(), os.path.join(cfg.MODEL_DIR, f"critic_epoch_{epoch+1}.pth"))        # 保存判别器权重 / Save critic weights
            
            # 可视化一个多资产样本 / Visualize a multi-asset sample
            generator.eval()  # 切换到评估模式 / Switch to eval mode
            with torch.no_grad():
                sample_noise = torch.randn(1, cfg.NOISE_DIM, device=cfg.DEVICE)  # 生成单个噪声 / Generate single noise
                sample_idx = np.random.randint(0, len(dataset))  # 随机选择一个样本索引 / Randomly select a sample index
                sample_real_path, sample_cond = dataset[sample_idx]  # 获取真实路径和条件 / Get real path and condition
                sample_cond = sample_cond.unsqueeze(0).to(cfg.DEVICE)  # 扩展条件维度并移动到设备 / Unsqueeze and move condition to device
                generated_path, _ = generator(sample_noise, sample_cond)  # 生成路径 / Generate path
                generated_path = generated_path.cpu().numpy().squeeze()  # 转为numpy并去除多余维度 / Convert to numpy and squeeze
                sample_real_path = sample_real_path.numpy().squeeze()    # 转为numpy并去除多余维度 / Convert to numpy and squeeze
                
                fig, axes = plt.subplots(cfg.NUM_ASSETS, 1, figsize=(14, 4 * cfg.NUM_ASSETS), sharex=True)
                if cfg.NUM_ASSETS == 1: axes = [axes]
                fig.suptitle(f"Epoch {epoch+1} - Multivariate Path Comparison", fontsize=16)  # 图标题 / Figure title
                
                for i in range(cfg.NUM_ASSETS):
                    axes[i].plot(sample_real_path[:, i], label=f"Real Path", color='blue', linestyle='--')  # 绘制真实路径 / Plot real path
                    axes[i].plot(generated_path[:, i], label=f"Generated Path", color='red')                # 绘制生成路径 / Plot generated path
                    axes[i].set_title(f"Asset: {cfg.ASSETS[i]}")  # 设置资产标题 / Set asset title
                    axes[i].legend()
                    axes[i].grid(True, linestyle='--')
                    
                plt.xlabel("Time Step (Trading Day)")  # x轴标签 / x-axis label
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(os.path.join(cfg.FIGURE_DIR, f"epoch_{epoch+1}_comparison.png"))  # 保存图片 / Save figure
                plt.close(fig)
            generator.train()  # 切换回训练模式 / Switch back to train mode

    total_time = (time.time() - train_start_time) / 60  # 计算总训练时间（分钟）/ Calculate total training time (minutes)
    log_and_print(f"--- Training finished! Total time: {total_time:.2f} minutes ---")  # 日志：训练结束 / Log: Training finished

if __name__ == "__main__":
    main()  # 程序入口 / Entry point