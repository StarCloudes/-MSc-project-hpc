CXX        = mpic++
CXXFLAGS   = -std=c++17 -Wall -O2 -Iinclude -I/opt/homebrew/include
LDFLAGS    = -L/opt/homebrew/lib -lboost_math_c99
SRC_DIR    = src
BUILD_DIR  = build

# Object files shared by both versions
OBJS_COMMON = \
    $(BUILD_DIR)/cholesky.o \
    $(BUILD_DIR)/copula_transform.o \
    $(BUILD_DIR)/generate_portfolio.o \
    $(BUILD_DIR)/monte_carlo.o \
    $(BUILD_DIR)/simulate_path.o \
    $(BUILD_DIR)/simulator.o \
    $(BUILD_DIR)/histogram.o \
    $(BUILD_DIR)/broadcast_utils.o

# Default target
all: monte_carlo_mpi_gather monte_carlo_mpi_reduce

# === Gather version ===
monte_carlo_mpi_gather: $(OBJS_COMMON) $(BUILD_DIR)/main_mpi_gather.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# === Reduce version ===
monte_carlo_mpi_reduce: $(OBJS_COMMON) $(BUILD_DIR)/main_mpi_reduce.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# === Compile all .cpp files ===
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# === Clean ===
clean:
	rm -rf $(BUILD_DIR) monte_carlo_mpi_gather monte_carlo_mpi_reduce

.PHONY: all clean