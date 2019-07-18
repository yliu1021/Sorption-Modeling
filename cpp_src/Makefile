CXX = g++
CPPFLAGS = -Wall -Wextra -Wc++11-extensions
sources = constants.h fast_dft_main.cpp fast_dft.cpp Makefile naive_random.cpp helpers.cpp

default: fast_dft

all: fast_dft naive_random

fast_dft: constants.h fast_dft.cpp fast_dft_main.cpp
	$(CXX) $(CPPFLAGS) -o fast_dft fast_dft_main.cpp

naive_random: constants.h fast_dft.cpp helpers.cpp naive_random.cpp
	$(CXX) $(CPPFLAGS) -o naive_random naive_random.cpp

# dist:
# 	tar -czf out.tar.gz $(sources)

clean:
	rm -f *.o fast_dft naive_random