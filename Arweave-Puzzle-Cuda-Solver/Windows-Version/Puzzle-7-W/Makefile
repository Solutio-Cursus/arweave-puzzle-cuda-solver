# Compiler
NVCC = nvcc

# Compiler flags
# -O3 for maximum optimization
# -std=c++11 as the code uses C++11 features
# -arch=sm_XX might need to be adjusted for your GPU. sm_61 (Pascal), sm_75 (Turing), sm_86 (Ampere)
# Leave it out for nvcc to choose a suitable architecture, or set it for maximum performance.
NVCC_FLAGS = -O3 -std=c++11

# Name of the final program
TARGET = ar-brute

# Source files
SOURCES = combined_main.cu

all: $(TARGET)

$(TARGET): $(SOURCES) crypto_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)