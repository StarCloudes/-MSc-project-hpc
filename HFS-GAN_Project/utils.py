# utils.py
# ==============================================================================
#  存放项目通用的辅助函数
# ==============================================================================

import os
from config import LOG_FILE, FIGURE_DIR, MODEL_DIR, REPORT_DIR

def log_and_print(message, log_file=LOG_FILE):
    """
    一个辅助函数，同时向控制台打印信息并记录到日志文件。
    A helper function to print to console and log to file simultaneously.
    """
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")

def setup_directories():
    """
    创建所有需要的输出目录。
    Create all necessary output directories.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    log_and_print("--- Project directories are set up. ---")

