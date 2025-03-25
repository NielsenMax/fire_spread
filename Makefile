CXX = g++
CXXFLAGS = -Wall -Wextra -Werror
OPTFLAGS = -O1
INCLUDE = -I./src
CXXCMD = $(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDE)
BINARY_NAME = burned_probabilities_data

headers = $(wildcard ./src/*.hpp)
sources = $(wildcard ./src/*.cpp)
objects_names = $(sources:./src/%.cpp=%)
objects = $(objects_names:%=./src/%.o)

mains = graphics/burned_probabilities_data graphics/fire_animation_data

all: $(mains)

%.o: %.cpp $(headers)
	$(CXXCMD) -c $< -o $@

$(mains): %: %.cpp $(objects) $(headers)
	$(CXXCMD) $< $(objects) -o $(dir $@)$(BINARY_NAME)

data.zip:
	wget https://cs.famaf.unc.edu.ar/~nicolasw/data.zip

data: data.zip
	unzip data.zip

clean:
	rm -f $(objects) graphics/$(BINARY_NAME)

.PHONY: all clean
