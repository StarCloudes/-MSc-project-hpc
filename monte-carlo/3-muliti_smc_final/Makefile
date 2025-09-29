# --- 编译器和编译选项 ---
# CXX: 使用 mpicc 作为 C 编译器 (根据您的要求修改)
# CXXFLAGS: 编译时传递给编译器的标志
#   -fopenmp:  *** 核心修复: 在编译阶段启用 OpenMP 支持 ***
CXX        = mpicc
CXXFLAGS   = -std=c++17 -Wall -O3 -g -fopenmp \
  -Iinclude \
  -I/opt/homebrew/include \
  -I/home/support/rl8/spack/0.21.2/spack/opt/spack/linux-rocky8-x86_64_v3/gcc-8.5.0/anaconda3-2023.09-0-jorvdessabbzrfz4boew6bk35wfjh5gg/include

# LDFLAGS: 链接时传递给链接器的标志
#   -fopenmp:  *** 核心修复: 链接 OpenMP 运行时库 (libgomp) ***
LDFLAGS    = -fopenmp \
  -L/home/support/rl8/spack/0.21.2/spack/opt/spack/linux-rocky8-x86_64_v3/gcc-8.5.0/anaconda3-2023.09-0-jorvdessabbzrfz4boew6bk35wfjh5gg/lib \
  -lboost_math_c99 \
  -lstdc++ \
  -lm


# --- 目录定义 ---
SRC_DIR    = src
BUILD_DIR  = build

# --- 最终模型所需的目标文件 (.o) ---
# 注意: 确保所有需要的 .cpp 文件对应的 .o 文件都在这里列出
OBJS_FINAL_MODEL = \
    $(BUILD_DIR)/generate_portfolio.o \
    $(BUILD_DIR)/histogram.o \
    $(BUILD_DIR)/broadcast_utils.o \
    $(BUILD_DIR)/cholesky.o \
    $(BUILD_DIR)/student_t_sampler.o \
    $(BUILD_DIR)/multi_factor_simulator.o \
    $(BUILD_DIR)/gan_preprocess.o \
    $(BUILD_DIR)/monte_carlo_final.o

# --- 默认目标 ---
all: simulation

# --- 构建最终的可执行文件 ---
# 将所有目标文件作为依赖项
simulation: $(OBJS_FINAL_MODEL)
	@echo "==> Linking to create final executable: simulation"
	$(CXX) -o $@ $^ $(LDFLAGS)

# --- 通用编译规则 ---
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	@echo "==> Compiling $< to $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- 清理命令 ---
clean:
	@echo "==> Cleaning up build files..."
	rm -rf $(BUILD_DIR) simulation

# --- 伪目标 ---
.PHONY: all clean
