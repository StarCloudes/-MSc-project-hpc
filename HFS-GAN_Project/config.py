# config.py
# ==============================================================================
#  v24 (Final Solution) - 引入直接波动率匹配损失，并重新平衡所有权重
#  v24 (最终方案) - Introduce direct volatility matching loss and rebalance all weights
# ==============================================================================

import torch

# --- 1. 路径与设备配置 ---
# 1. Path and device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))  # 选择设备：优先GPU，其次Apple MPS，否则CPU / Select device: prefer GPU, then Apple MPS, else CPU
LOG_FILE = "training_log_multivariate_v24.txt"  # 日志文件路径 / Log file path
FIGURE_DIR = "results/plots_v24"                # 图像保存目录 / Directory for saving plots
MODEL_DIR = "saved_models_v24"                  # 模型保存目录 / Directory for saving models
REPORT_DIR = "results/reports_v24"              # 报告保存目录 / Directory for saving reports

# --- 2. 数据与模型核心参数 ---
# 2. Data and core model parameters
ASSETS = ['^GSPC', '^IXIC', 'GC=F']             # 资产列表 / List of assets
NUM_ASSETS = len(ASSETS)                        # 资产数量 / Number of assets
START_DATE = "2000-01-01"                       # 数据起始日期 / Data start date
END_DATE = "2023-12-31"                         # 数据结束日期 / Data end date
SEQ_LEN = 64                                    # 输入序列长度 / Input sequence length

# 模型维度 / Model dimensions
D_MODEL = 128                                   # Transformer隐藏层维度 / Transformer hidden dimension
N_HEAD = 4                                      # 多头注意力头数 / Number of attention heads
GEN_ENCODER_LAYERS = 2                          # 生成器编码层数 / Generator encoder layers
CRITIC_ENCODER_LAYERS = 4                       # 判别器编码层数 / Critic encoder layers
DROPOUT = 0.1                                   # Dropout比例 / Dropout rate
NOISE_DIM = 100                                 # 噪声向量维度 / Noise vector dimension
COND_DIM = NUM_ASSETS * 2                       # 条件向量维度 / Conditional vector dimension

# --- 3. 训练过程参数 ---
# 3. Training parameters
EPOCHS = 10000                                  # 训练轮数 / Number of training epochs
BATCH_SIZE = 128                                # 批次大小 / Batch size
LR_G = 2e-5                                     # 生成器学习率 / Generator learning rate
LR_C = 2e-4                                     # 判别器学习率 / Critic learning rate
BETA_1 = 0.0                                    # Adam优化器beta1参数 / Adam optimizer beta1
BETA_2 = 0.9                                    # Adam优化器beta2参数 / Adam optimizer beta2
N_CRITIC = 5                                    # 每训练生成器一次，判别器训练次数 / Critic steps per generator step
SAVE_INTERVAL = 500                             # 模型保存间隔 / Model save interval

# --- 4. 损失函数权重 ---
# 4. Loss function weights 
LAMBDA_GP = 10.0                                # 梯度惩罚权重 / Gradient penalty weight
LAMBDA_VOL = 400.0                              # 波动率损失权重 / Volatility loss weight
LAMBDA_CORR = 500.0                             # 相关性损失权重 / Correlation loss weight
LAMBDA_STYLE = 50.0                             # 风格损失权重 / Style loss weight

# --- 5. 评估脚本参数 ---
# 5. Evaluation script parameters
EVAL_MODEL_PATH = "saved_models_v24/generator_epoch_10000.pth"  # 评估用生成器模型路径 / Path to generator model for evaluation
EVAL_SAMPLE_SIZE = 5000                                         # 评估采样数量 / Number of samples for evaluation

# --- 6. 蒙特卡洛场景生成参数  ---
# 6. Monte Carlo scenario generation parameters 
SCENARIO_GENERATION_SIZE = 200000               # 生成场景数量 / Number of scenarios to generate
SCENARIO_OUTPUT_FILE = "gan_scenarios.csv"      # 场景输出文件名 / Output file name for scenarios