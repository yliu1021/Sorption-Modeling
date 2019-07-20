CXX = c++
CXXFLAGS = -Wall -Wextra -std=c++14
sources = constants.h fast_dft_main.cpp fast_dft.cpp Makefile naive_random.cpp helpers.cpp

all: fast_dft evolution

fast_dft: fast_dft.o helpers.o
	$(CXX) $(CXXFLAGS) -o fast_dft fast_dft.o helpers.o

fast_dft.o: fast_dft.cpp helpers.h constants.h

evolution: evolution.o helpers.o
	$(CXX) $(CXXFLAGS) -o evolution evolution.o helpers.o

evolution.o: evolution.cpp helpers.h constants.h

helpers.o: helpers.h

# dist:
# 	tar -czf out.tar.gz $(sources)

clean:
	rm -f *.o fast_dft naive_random evolution