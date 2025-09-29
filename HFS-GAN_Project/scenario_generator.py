# scenario_generator.py
# ==============================================================================
#  功能: 加载一个训练好的HFS-GAN生成器，并为蒙特卡洛模拟生成大量的
#        系统性风险因子场景，最终保存为CSV文件。 The path to the output CSV file.
# ==============================================================================

import torch
import numpy as np
import os
import sys

# 导入自定义模块
import config as cfg
from model import Generator
from data_loader import MultivariateRealFinancialDataset as RealFinancialDataset
from utils import log_and_print, setup_directories

def generate_scenarios_for_monte_carlo(generator_model, dataset, num_scenarios, output_filename):
    """
    使用训练好的生成器生成大量的系统性风险因子向量，并保存为CSV文件。

    Args:
        generator_model (nn.Module): The trained generator model.
        dataset (Dataset): The dataset object to source a representative condition from.
        num_scenarios (int): The number of scenario vectors to generate.
        output_filename (str): The path to the output CSV file.
    """
    log_and_print(f"--- Generating {num_scenarios} Scenarios for Monte Carlo Simulation ---")
    generator_model.eval()
    
    # 使用数据集中最后一天的市场情况作为生成所有场景的条件
    # This represents the most recent market state available.
    _, sample_cond = dataset[len(dataset)-1]
    sample_cond = sample_cond.unsqueeze(0).to(cfg.DEVICE)
    
    all_scenarios = []
    
    # To be memory-efficient for very large numbers of scenarios, generate in batches
    # and append to the list.
    batch_size = 512 # A reasonable batch size for generation
    num_batches = (num_scenarios + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_scenarios - i * batch_size)
            if current_batch_size <= 0:
                break
            
            if (i + 1) % 20 == 0:
                log_and_print(f"  ... generated {(i + 1) * batch_size}/{num_scenarios} scenarios")
                
            noise = torch.randn(current_batch_size, cfg.NOISE_DIM, device=cfg.DEVICE)
            # We need to expand the single condition to match the batch size of the noise
            expanded_cond = sample_cond.expand(current_batch_size, -1)
            
            _, fake_log_returns = generator_model(noise, expanded_cond)
            
            # The C++ MC simulation requires a single vector of shocks per path.
            # We take the log returns from the first day of the generated sequence 
            # as the representative systemic shock vector for this simulation path.
            scenario_vectors = fake_log_returns[:, 0, :].cpu().numpy()
            all_scenarios.append(scenario_vectors)
            
    # Convert list of batches to a single NumPy array and save to CSV
    scenarios_array = np.concatenate(all_scenarios, axis=0)
    np.savetxt(output_filename, scenarios_array, delimiter=",", fmt='%.8f')
    log_and_print(f"Scenarios successfully saved to {output_filename}. Shape: {scenarios_array.shape}")

def main():
    """
    主执行函数
    """
    # --- 1. 设置与模型加载 ---
    setup_directories()
    log_and_print("--- Starting Monte Carlo Scenario Generation ---")
    log_and_print(f"Loading model from: {cfg.EVAL_MODEL_PATH}")

    generator = Generator().to(cfg.DEVICE)
    try:
        generator.load_state_dict(torch.load(cfg.EVAL_MODEL_PATH, map_location=cfg.DEVICE))
    except Exception as e:
        log_and_print(f"Error loading model: {e}")
        sys.exit()
    
    # --- 2. 准备数据集以获取条件 ---
    log_and_print("Loading dataset to get a representative market condition...")
    dataset = RealFinancialDataset(
        assets=cfg.ASSETS, 
        start_date=cfg.START_DATE, 
        end_date=cfg.END_DATE, 
        seq_len=cfg.SEQ_LEN
    )

    # --- 3. 调用场景生成函数 ---
    generate_scenarios_for_monte_carlo(
        generator_model=generator,
        dataset=dataset,
        num_scenarios=cfg.SCENARIO_GENERATION_SIZE,
        output_filename=cfg.SCENARIO_OUTPUT_FILE
    )
    
    log_and_print("--- Scenario generation finished successfully. ---")

if __name__ == "__main__":
    main()
