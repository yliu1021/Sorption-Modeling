CXX = c++
CXXFLAGS = -Wall -Wextra -std=c++17

vpath %.cpp src
vpath %.h src
TARGETS := fast_dft evolution

all: fast_dft evolution

fast_dft: fast_dft.o helpers.o
	$(CXX) $(CXXFLAGS) -o fast_dft fast_dft.o helpers.o

evolution: evolution.o helpers.o
	$(CXX) $(CXXFLAGS) -o evolution evolution.o helpers.o

fast_dft.o: fast_dft.cpp helpers.h constants.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

evolution.o: evolution.cpp helpers.h constants.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

helpers.o: helpers.h

.PHONY: clean
clean:
	rm -f *.o $(TARGETS)