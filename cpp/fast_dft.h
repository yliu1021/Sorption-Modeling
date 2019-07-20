// #ifndef FAST_DFT_H_
// #define FAST_DFT_H_

// #include <iostream>
// #include <iomanip>
// #include <fstream>
// #include <array>
// #include <string>
// #include <thread>

// #include <cstring>
// #include <cmath>
// #include <cstdio>
// #include <cstdlib>

// #include "constants.h"

// using namespace std;

// #define KB 0.0019872041
// #define T 298.0
// #define Y 1.5
// #define TC 647.0
// const double BETA = 1 / (KB * T);
// const double MUSAT = -2.0 * KB * TC;
// const double C = 4.0;
// const double WFF = -2.0 * MUSAT / C;

// int NL[N_SQUARES + 1][N_SQUARES];

// // setup code for run_dft
// void setup_NL();

// /*
// Computes and returns the density of a grid
// 	grid: a N_SQUARES long array of doubles
// */
// std::array<double, N_ITER + 1> run_dft(std::array<double, N_SQUARES> grid);

// #endif // FAST_DFT_H_