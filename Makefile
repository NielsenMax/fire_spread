CXX = nvcc
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_60 --use_fast_math -Xcompiler -Wall,-Wextra,-fopenmp
OPTFLAGS = -O3
INCLUDE = -I./src -I/usr/include
NVCCCMD = $(NVCC) $(NVCCFLAGS) $(OPTFLAGS) $(INCLUDE)
BINARY_NAME ?= burned_probabilities_data
MAIN_FILE ?= burned_probabilities_data

headers = $(wildcard ./src/*.hpp) $(wildcard ./src/*.cuh)
sources = $(wildcard ./src/*.cpp) $(wildcard ./src/*.cu)

# Generate object file names in src/ directory
objects = $(addprefix src/, $(addsuffix .o, $(basename $(notdir $(sources)))))

mains = graphics/burned_probabilities_data graphics/fire_animation_data

# Default target builds both mains
all: $(mains)

src/%.o: src/%.cpp $(headers)
	$(NVCCCMD) -c $< -o $@

src/%.o: src/%.cu $(headers)
	$(NVCCCMD) -c $< -o $@

$(mains): %: %.cpp $(objects) $(headers)
	$(NVCCCMD) $< $(objects) -o $@ -lcudart

# Build a specific binary with custom name
specific: graphics/$(MAIN_FILE).cpp $(objects) $(headers)
	$(NVCCCMD) $< $(objects) -o binaries/$(BINARY_NAME) -lcudart

data.zip:
	wget https://cs.famaf.unc.edu.ar/~nicolasw/data.zip

data: data.zip
	unzip data.zip

clean:
	rm -f $(objects) $(mains) graphics/$(BINARY_NAME)

.PHONY: all clean specific
