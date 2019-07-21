#ifndef HELPERS_H_
#define HELPERS_H_

#include <array>
#include <iterator>
#include <iostream>
#include <vector>

#include "constants.h"

// ============================================================================
// I/O
// ============================================================================

/*
  Loads a grid from a grid_####.csv file.
*/
std::array<double, N_SQUARES> load_grid(std::istream &grid_file);
std::array<double, N_SQUARES> load_grid(const std::string &path);
std::array<double, N_ITER+1> load_density(const std::string &path);
void write_density(std::array<double, N_ITER+1> density, std::ostream &density_file);
bool write_density(std::array<double, N_ITER+1> density, const std::string &path);

// ============================================================================
// Grid Mutators
// ============================================================================

/*
  Randomly toggle one square in the grid.
 */
void toggle_random(std::array<double, N_SQUARES> &grid);

// ============================================================================
// Math Helpers
// ============================================================================

#define EPSILON 0.0000001

// Normalize the values in an array to between 0 and 1
void normalizeVec(std::vector<double> vec);

/*
  Clip (limit) the values in an array.
  Given an interval, values outside the interval are clipped to the interval 
  edges. For example, if an interval of [0, 1] is specified, values smaller than 
  0 become 0, and values larger than 1 become 1. 
*/
// void clip(std::array<double, N_ADSORP+1> a, const double a_min, const double a_max);

// ============================================================================
// Cost functions
// ============================================================================

double mean_abs_error(const std::array<double, N_ADSORP+1> y_true, const std::array<double, N_ITER+1> y_pred);

// double kullback_leibler_divergence(const std::array<double, N_ADSORP+1> y_true, const std::array<double, N_ITER+1> y_pred);

// ============================================================================
// Target curves
// ============================================================================

std::array<double, N_ADSORP+1> linear_curve();

// ============================================================================
// DFT Simulation
// ============================================================================

#define KB 0.0019872041
#define T 298.0
#define Y 1.5
#define TC 647.0
const double BETA = 1 / (KB * T);
const double MUSAT = -2.0 * KB * TC;
const double C = 4.0;
const double WFF = -2.0 * MUSAT / C;

/*
  Set up DFT simulation. Run once before using run_dft.
 */
void setup_NL();

/*
  Compute and return the sorption curve of a grid.
  @param grid a N_SQUARES long array of doubles
  @return a N_ITER+1 long array of doubles
 */
std::array<double, N_ITER + 1> run_dft(std::array<double, N_SQUARES> grid);

#endif // HELPERS_H_