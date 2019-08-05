//
//  Helpers.h
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#ifndef Helpers_h
#define Helpers_h

#include <array>
#include <iostream>
#include <iterator>
#include <vector>
#include <thread>

#include "Constants.h"

// ============================================================================
// I/O
// ============================================================================

/*
 * Loads a grid from a grid_####.csv file.
 */
std::array<double, N_SQUARES> load_grid(std::istream &grid_file);
std::array<double, N_SQUARES> load_grid(const std::string &path);
std::array<double, N_ITER+1> load_density(const std::string &path);
void write_grid(std::array<double,N_SQUARES> grid, std::ostream &grid_file);
bool write_grid(std::array<double,N_SQUARES> grid, const std::string &path);
void write_density(std::array<double, N_ITER+1> density, std::ostream &density_file);
bool write_density(std::array<double, N_ITER+1> density, const std::string &path);

// ============================================================================
// Grid Helpers
// ============================================================================

/*
 * Returns a randomly generated grid
 */
std::array<double, N_SQUARES> random_grid();

/*
 * Randomly toggle one square in the grid.
 */
void toggle_random(std::array<double, N_SQUARES> &grid);

// ============================================================================
// Math Helpers
// ============================================================================

#define EPSILON 0.0000001

// Rescale the values of a vector into a range of [0, 1]
void normalizeVec(std::vector<double> &v);

// Rescale the values of a vector to have a mean of 0 and a stdev of 1
void standardizeVec(std::vector<double> &v);

/*
 * Clip (limit) the values in an array.
 * Given an interval, values outside the interval are clipped to the interval
 * edges. For example, if an interval of [0, 1] is specified, values smaller than
 * 0 become 0, and values larger than 1 become 1.
 */
// void clip(std::array<double, N_ADSORP+1> a, const double a_min, const double a_max);

// ============================================================================
// Cost functions
// ============================================================================

double mean_abs_error(const std::array<double, N_ADSORP+1> &y_true, const std::array<double, N_ITER+1> &y_pred);

// double kullback_leibler_divergence(const std::array<double, N_ADSORP+1> y_true, const std::array<double, N_ITER+1> y_pred);

// ============================================================================
// Target curves
// ============================================================================

std::array<double, N_ADSORP+1> linear_curve();
std::array<double, N_ADSORP+1> heaviside_step_function(double c = 0.5);
std::array<double, N_ADSORP+1> step_function(std::vector<double> step_height, std::vector<double> step_size);
std::array<double, N_ADSORP+1> circular_curve(double radius=1, bool concave_up=true);

// ============================================================================
// DFT Simulation
// ============================================================================

constexpr double KB = 0.0019872041;
constexpr double T = 298.0;
constexpr double Y = 1.5;
constexpr double TC = 647.0;
constexpr double BETA = 1 / (KB * T);
constexpr double MUSAT = -2.0 * KB * TC;
constexpr double C = 4.0;
constexpr double WFF = -2.0 * MUSAT / C;

/*
 * Set up DFT simulation. Run once before using run_dft.
 */
void setup_NL();

/*
 * Compute and return the sorption curve of a grid.
 * @param grid a N_SQUARES long array of doubles
 * @return a N_ITER+1 long array of doubles
 */
std::array<double, N_ITER + 1> run_dft(std::array<double, N_SQUARES> grid);
std::array<double, N_ITER + 1> run_dft_fast(std::array<double, N_SQUARES> grid);

/*
 * These two functions are pretty horrific atm, but they work. Might come back and improve if I have time
 * TODO:
 */
//template <typename Iter1, typename Iter2, typename Iter3>
//void get_costs_thread(int threadID,
//                      const std::array<double,N_ADSORP+1> &target_curve,
//                      const CostFunction &cost_func,
//                      Iter1 grids_begin,
//                      Iter2 grids_end,
//                      Iter3 costs_begin) {
//    for (int i = threadID; (i+grids_begin)-grids_begin < (grids_end-grids_begin); i += NUM_THREADS) {
//        *(costs_begin+i) = (*cost_func)(target_curve, run_dft(*(grids_begin+i)));
//    }
//}
//
//template <typename Iter1, typename Iter2, typename Iter3>
//void get_costs(const std::array<double,N_ADSORP+1> &target_curve,
//               const CostFunction &cost_func,
//               Iter1 grids_begin,
//               Iter2 grids_end,
//               Iter3 costs_begin) {
//
//    std::array<std::thread, NUM_THREADS> threads;
//    for (int i = 0; i < threads.size(); ++i) {
//        threads[i] = std::thread(get_costs_thread<Iter1, Iter2, Iter3>, i, ref(target_curve), ref(cost_func), grids_begin, grids_end, costs_begin);
//    }
//    for (int i = 0; i < threads.size(); ++i) {
//        threads[i].join();
//    }
//}




#endif /* Helpers_h */
