#ifndef HELPERS_H_
#define HELPERS_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>
#include <string>
#include <thread>
#include <math.h>
#include <stdlib.h>
#include <iterator>
#include <vector>

#include "constants.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "constants.h"

using namespace std;

/*
Loads a grid from a grid_####.csv file
*/
std::array<double, N_SQUARES> load_grid(istream &grid_file);
std::array<double, N_SQUARES> load_grid(const string &path);
std::array<double, N_ITER+1> load_density(const string &path);
void write_density(std::array<double, N_ITER+1> density, ostream &density_file);
bool write_density(std::array<double, N_ITER+1> density, const string &path);


#define EPSILON 0.0000001

void toggle_random(std::array<double, N_SQUARES> &grid);

// Normalize the values in an array to between 0 and 1
void normalizeArr(std::vector<double>::iterator begin, std::vector<double>::iterator end, double min, double max);

void normalizeArr(double* piStart, double* piLast, double min, double max);

// /*
// Clip (limit) the values in an array.
// Given an interval, values outside the interval are clipped to the interval 
// edges. For example, if an interval of [0, 1] is specified, values smaller than 
// 0 become 0, and values larger than 1 become 1. 
// */
// void clip(double a[], double a_min, double a_max, unsigned short len=N_ADSORP+1);

void clip(std::array<double, N_ADSORP+1> a, const double a_min, const double a_max);

// double kullback_leibler_divergence(const std::array<double, N_ADSORP+1> y_true, const std::array<double, N_ITER+1> y_pred);

double mean_abs_error(const std::array<double, N_ADSORP+1> y_true, const std::array<double, N_ITER+1> y_pred);

// double kullback_leibler_divergence(double y_true[], double y_pred[], unsigned short len=N_ADSORP+1);

std::array<double, N_ADSORP+1> linear_curve();



#define KB 0.0019872041
#define T 298.0
#define Y 1.5
#define TC 647.0
const double BETA = 1 / (KB * T);
const double MUSAT = -2.0 * KB * TC;
const double C = 4.0;
const double WFF = -2.0 * MUSAT / C;


// setup code for run_dft
void setup_NL();

/*
Computes and returns the density of a grid
	grid: a N_SQUARES long array of doubles
*/
std::array<double, N_ITER + 1> run_dft(std::array<double, N_SQUARES> grid);

#endif // HELPERS_H_