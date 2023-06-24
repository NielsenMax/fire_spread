CXX = g++
CXXFLAGS = -Wall -Wextra -Werror
CPPOPTFLAGS = -O2
INCLUDE = -I./src
CXXCMD = $(CXX) $(CXXFLAGS) $(CPPOPTFLAGS) $(INCLUDE)

headers = $(wildcard ./src/*.hpp)
sources = $(wildcard ./src/*.cpp)
objects_names = $(sources:./src/%.cpp=%)
objects = $(objects_names:%=./src/%.o)

mains = graphics/burned_probabilities_data graphics/fire_animation_data

all: $(mains)

%.o: %.cpp $(headers)
	$(CXXCMD) -c $< -o $@

$(mains): %: %.cpp $(objects) $(headers)
	$(CXXCMD) $< $(objects) -o $@

clean:
	rm -f $(objects) $(mains)

.PHONY: all clean
