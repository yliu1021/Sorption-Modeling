CXX = c++
CXXFLAGS = -Wall -Wextra -std=c++17

vpath %.cpp src
INCLUDE := -Iinclude/
TARGETS := fast_dft evolution

all: fast_dft evolution

fast_dft: fast_dft.o helpers.o
	$(CXX) $(CXXFLAGS) -o fast_dft fast_dft.o helpers.o

evolution: evolution.o helpers.o
	$(CXX) $(CXXFLAGS) -o evolution evolution.o helpers.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

.PHONY: clean
clean:
	rm -f *.o $(TARGETS)