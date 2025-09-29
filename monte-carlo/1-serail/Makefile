CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

INCLUDES = -Iinclude -I/opt/homebrew/include
LIBS = -L/opt/homebrew/lib -lboost_math_c99

SRC_DIR = src
OBJ_DIR = build
TARGET = monte_carlo_credit_risk

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SOURCES))

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJECTS) -o $@ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean