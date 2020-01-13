#ifndef CONSTANTS_H_
#define CONSTANTS_H_

constexpr int GRID_SIZE = 20;
constexpr int N_SQUARES = (GRID_SIZE * GRID_SIZE);
constexpr int N_ITER = 80;
constexpr int N_ADSORP = 40;
constexpr double STEP_SIZE = 0.025;

typedef std::array<double, N_SQUARES> Grid;

#endif // CONSTANTS_H_
