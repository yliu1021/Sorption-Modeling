CXX = c++
CXXFLAGS = -Wall -Wextra -std=c++17 -O3 -funroll-loops -march=native

vpath %.cpp src
vpath %.h src
TARGETS := fast_dft evolution naive_min swarm benchmark

all: fast_dft evolution naive_min benchmark

fast_dft: fast_dft.o helpers.o
	$(CXX) $(CXXFLAGS) -o fast_dft fast_dft.o helpers.o

evolution: evolution.o helpers.o
	$(CXX) $(CXXFLAGS) -o evolution evolution.o helpers.o

benchmark: benchmark_dft.o helpers.o
	$(CXX) $(CXXFLAGS) -o benchmark benchmark_dft.o helpers.o

naive_min: naive_min.o helpers.o
	$(CXX) $(CXXFLAGS) -o naive_min naive_min.o helpers.o

swarm: swarm.o helpers.o
	$(CXX) $(CXXFLAGS) -o swarm swarm.o helpers.o
	
benchmark_dft.o: benchmark_dft.cpp helpers.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

fast_dft.o: fast_dft.cpp helpers.h constants.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

evolution.o: evolution.cpp helpers.h constants.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

naive_min.o: naive_min.cpp helpers.h constants.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

swarm.o: swarm.cpp helpers.h constants.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

helpers.o: helpers.h

.PHONY: clean
clean:
	rm -f *.o $(TARGETS)