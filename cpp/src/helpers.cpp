#include "helpers.h"

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>
#include <string>
#include <vector>
#include <iterator>
#include <numeric>
#include <valarray>

#include <math.h>
#include <stdlib.h>

#include "constants.h"

using namespace std;

// ============================================================================
// I/O
// ============================================================================

array<double,N_SQUARES> load_grid(istream &grid_file) {
    array<double,N_SQUARES> grid;
	int pos = 0;
	string line;
	while (getline(grid_file, line) && pos < N_SQUARES) {
		int num_digits = 0;
		for (char c : line) {
			if (c == '1') {
				grid[pos++] = 1;
				++num_digits;
			} else if (c == '0') {
				grid[pos++] = 0;
				++num_digits;
			}
		}
		if (num_digits != GRID_SIZE) {
			cerr << "File improperly formatted. Got " << num_digits << " digits in a row" << endl;
            exit(1);
		}
	}
	return grid;
}

array<double,N_SQUARES> load_grid(const string &path) {
    ifstream grid_file;
    grid_file.open(path);
	if (!grid_file.is_open()) {
		cerr << "File " << path << " doesn't exist" << endl;
        exit(1);
	}
	
	array<double,N_SQUARES> grid = load_grid(grid_file);
	
	grid_file.close();
	return grid;
}

array<double,N_ITER+1> load_density(const string &path) {
	ifstream density_file;
	density_file.open(path);
	if (!density_file.is_open()) {
		cerr << "File " << path << " doesn't exist" << endl;
        exit(1);
	}
	
    array<double,N_ITER+1> densities;
	string line;
	while (getline(density_file, line)) {
		if (line[0] == ',') continue;
		size_t pos;
		int ind = stoi(line, &pos);
		string den_str = line.substr(pos + 1);
		double density = stod(den_str);
		densities[ind] = density;
	}
	
	density_file.close();
	return densities;
}

void write_grid(array<double,N_SQUARES> grid, ostream &grid_file) {
	for (int i = 0; i < GRID_SIZE; ++i) {
		for (int j = 0; j < GRID_SIZE; ++j) {
			grid_file << grid[i*GRID_SIZE+j] << ",";
		}
		grid_file << endl;
	}
}

bool write_grid(array<double,N_SQUARES> grid, const string &path) {
	ofstream grid_file;
	grid_file.open(path);
	if (!grid_file.is_open()) {
		cerr << "Could not open/create grid file" << endl;
		return false;
	}
	write_grid(grid, grid_file);
	grid_file.close();
	return true;
}

void write_density(array<double,N_ITER+1> density, ostream &density_file) {
	density_file << ",0" << endl;
	for (int i = 0; i <= N_ITER; ++i) {
		density_file << i << "," << setprecision(17) << density[i] << endl;
	}
}

bool write_density(array<double,N_ITER+1> density, const string &path) {
	ofstream density_file;
	density_file.open(path);
	if (!density_file.is_open()) {
		cerr << "Could not open/create density file" << endl;
		return false;
	}
	write_density(density, density_file);
	density_file.close();
	return true;
}

// ============================================================================
// Grid Mutators
// ============================================================================

void toggle_random(array<double, N_SQUARES> &grid) {
    int sq = rand() % N_SQUARES;
    grid[sq] = 1- grid[sq];
}

// ============================================================================
// Math Helpers
// ============================================================================

void normalizeVec(vector<double> &v) {
	double min = *min_element(v.begin(), v.end());
	double range = *max_element(v.begin(), v.end()) - min;
	for (auto &x : v) {
		x -= min;
		x /= range;
	}
}

void standardizeVec(vector<double> &v) {
	double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
	double mean =  sum / v.size();

	double accum = 0.0;
	std::for_each (std::begin(v), std::end(v), [&](const double d) {
		accum += (d - mean) * (d - mean);
	});
	double stdev = sqrt(accum / (v.size()-1));

	for (auto &x : v) {
		x -= mean;
		x /= stdev;
	}
}

/*
Clip (limit) the values in an array.
Given an interval, values outside the interval are clipped to the interval 
edges. For example, if an interval of [0, 1] is specified, values smaller than 
0 become 0, and values larger than 1 become 1. 
*/
// void clip(array<double, N_ADSORP+1> a, const double a_min, const double a_max) {
//     for (short i = 0; i < a.size(); ++i) { 
//         a[i] = max(a[i], a_min);
//         a[i] = min(a[i], a_max);
//     } 
// }

// ============================================================================
// Cost functions
// ============================================================================

double mean_abs_error(const array<double, N_ADSORP+1> &y_true, const array<double, N_ITER+1> &y_pred) {
    double mse = 0;
    for (short i = 0; i < N_ADSORP+1; ++i) {
        mse += abs(y_true[i] - y_pred[i]);
    }
    mse /= (N_ADSORP+1);
    return mse;
}

// double kullback_leibler_divergence(const array<double, N_ADSORP+1> y_true, const array<double, N_ITER+1> y_pred) {
//     // TODO: Something wrong-- NAN values
//     clip(y_true, EPSILON, 1);
//     clip(y_pred, EPSILON, 1);
//     double sum = 0;
//     for (short i = 0; i < N_ADSORP+1; ++i) {
//         cout << y_true[i] << "\t";
//         cout << y_pred[i] << "\t";
//         cout << y_true[i]/y_pred[i] << endl;
//         sum += y_true[i] * log(y_true[i] / y_pred[i]);
//     }
//     cout << "SUM: " << sum << endl;
//     return sum;
// }

// ============================================================================
// Target curves
// ============================================================================

array<double, N_ADSORP+1> linear_curve() {
    array<double, N_ADSORP+1> lin;
    double v = 0;
    for (short i = 0; i < N_ADSORP+1; ++i, v += STEP_SIZE) {
        lin[i] = v;
    }
    return lin;
}

array<double, N_ADSORP+1> heaviside_step_function(double c) {
    array<double, N_ADSORP+1> f;
	for (short i = 0; i <= N_ADSORP+1; ++i) {
		if (i*STEP_SIZE < c) { f[i] = 0; }
		else if (i*STEP_SIZE == c) { f[i] = 0.5; }
		else { f[i] = 1; }
	}
	return f;
}

// ============================================================================
// DFT Simulation
// ============================================================================

int NL[N_SQUARES + 1][N_SQUARES];
double muu_lookup[N_ITER + 1];

void setup_NL() {
	double r[2][N_SQUARES + 1];	
	double Lx = 0.0;
	r[0][0] = 0.0;
	for (int i = 1; i <= N_SQUARES; ++i) {
		r[0][i] = Lx;
		Lx += 1;
		if (i % GRID_SIZE == 0)
			Lx = 0;
	}
	double Ly = 0.0;
	r[1][0] = 0.0;
	for (int i = 1; i <= N_SQUARES; ++i) {
		r[1][i] = Ly;
		if (i % GRID_SIZE == 0)
			Ly += 1;
	}

	double rc = 1.01;
	double rc_square = rc * rc;
	int NN[N_SQUARES + 1];
	memset(NL, 0, N_SQUARES * (N_SQUARES + 1) * sizeof(int));
	memset(NN, 0, (N_SQUARES + 1) * sizeof(int));
	for (int i = 1; i < N_SQUARES; ++i) {
		for (int jj = i + 1; jj <= N_SQUARES; ++jj) {
			double r11 = r[0][jj] - r[0][i];
			double r12 = r[1][jj] - r[1][i];
			r11 = r11 - (int)(((double) r11 / GRID_SIZE) + 0.5) * GRID_SIZE;
			r12 = r12 - (int)(((double) r12 / GRID_SIZE) + 0.5) * GRID_SIZE;
			double d12_square = r11*r11 + r12*r12;
			if (d12_square <= rc_square) {
				NN[i] += 1;
				NN[jj] += 1;
				NL[i][NN[i]] = jj;
				NL[jj][NN[jj]] = i;
			}
		}
	}
	
	for (int jj = 0; jj <= N_ITER; ++jj) {
		double RH;
		double muu = -90;
		if (jj <= N_ADSORP) {
			RH = jj * STEP_SIZE;
		} else {
			RH = N_ADSORP*STEP_SIZE - (jj - N_ADSORP)*STEP_SIZE;
		}
		if (RH != 0.0) {
			muu = MUSAT + KB*T*log(RH);
		}
		muu_lookup[jj] = muu;
	}
}

array<double,N_ITER+1> run_dft(array<double,N_SQUARES> grid) {
    array<double,N_ITER+1> density;
	double r[2][N_SQUARES + 1];
	r[0][0] = 0.0;
	r[1][0] = 0.0;
	
	double Ntotal_pores = 0.0;
	for (int i = 1; i <= N_SQUARES; ++i) {
		double g = grid[i - 1];
		Ntotal_pores += g;
		r[0][i] = g;
	}
	if (Ntotal_pores < 0.1) {
		// no pores, return all 0's
		for (int i = 0; i <= N_ITER; ++i) {
			density[i] = 0.0;
		}
		return density;
	}
	
	for (int i = 1; i <= N_SQUARES; ++i) {
		r[1][i] = grid[i - 1];
	}
	
	for (int jj = 0; jj <= N_ITER; ++jj) {
		double muu = muu_lookup[jj];
		
		for (int c = 1; c < 100000000; ++c) {
            // vi = veff(r,muu,NL)
			double vi[N_SQUARES + 1];
			for (int i = 1; i <= N_SQUARES; ++i) {
				int a1 = NL[i][1];
				int a2 = NL[i][2];
				int a3 = NL[i][3];
				int a4 = NL[i][4];
				vi[i] = WFF * (r[1][a1] + Y * (1 - r[0][a1])) +
						WFF * (r[1][a2] + Y * (1 - r[0][a2])) +
						WFF * (r[1][a3] + Y * (1 - r[0][a3])) +
						WFF * (r[1][a4] + Y * (1 - r[0][a4])) +
						muu;
			}
			// rounew = rou(vi,r)
			double power_drou = 0.0;
			double rounew[N_SQUARES + 1];
			
			for (int i = 0; i <= N_SQUARES; ++i) {
				rounew[i] = r[0][i] / (1 + exp(-BETA * vi[i]));
			}
			for (int i = 0; i <= N_SQUARES; ++i) {
				double diff = rounew[i] - r[1][i];
				power_drou += diff * diff;
				r[1][i] = rounew[i];
			}
			if (power_drou < 1e-10 * N_SQUARES) {
				break;
			}
		}
		density[jj] = r[1][0];
		for (int i = 1; i <= N_SQUARES; ++i) {
			density[jj] += r[1][i];
		}
		density[jj] /= Ntotal_pores;
	}
	return density;
}

// This is a "faster"-ish dft but it's for experimental benchmarking.
// It's best to just use the normal run_dft function.
array<double,N_ITER+1> run_dft_fast(array<double,N_SQUARES> grid) {
    array<double,N_ITER+1> density;
	double r[2][N_SQUARES + 1];
	r[0][0] = 0.0;
	r[1][0] = 0.0;
	
	double Ntotal_pores = 0.0;
	for (int i = 1; i <= N_SQUARES; ++i) {
		double g = grid[i - 1];
		Ntotal_pores += g;
		r[0][i] = g;
	}
	if (Ntotal_pores < 0.1) {
		// no pores, return all 0's
		for (int i = 0; i <= N_ITER; ++i) {
			density[i] = 0.0;
		}
		return density;
	}
	
	for (int i = 1; i <= N_SQUARES; ++i) {
		r[1][i] = grid[i - 1];
	}
	
	for (int jj = 0; jj <= N_ITER; ++jj) {
		double muu = muu_lookup[jj];
		
		for (int c = 1; c < 100000000; ++c) {
            // vi = veff(r,muu,NL)
			double vi[N_SQUARES + 1];
			for (int i = 1; i <= N_SQUARES; ++i) {
				int a1 = NL[i][1];
				int a2 = NL[i][2];
				int a3 = NL[i][3];
				int a4 = NL[i][4];
				double vi1 = WFF * (r[1][a1] + Y * (1 - r[0][a1]));
				double vi2 = WFF * (r[1][a2] + Y * (1 - r[0][a2]));
				double vi3 = WFF * (r[1][a3] + Y * (1 - r[0][a3]));
				double vi4 = WFF * (r[1][a4] + Y * (1 - r[0][a4]));
				vi[i] = vi1 + vi2 + vi3 + vi4 + muu;
			}
			// rounew = rou(vi,r)
			double power_drou = 0.0;
			double rounew[N_SQUARES + 1];
			
			for (int i = 0; i <= N_SQUARES; ++i) {
				rounew[i] = r[0][i] / (1 + exp(-BETA * vi[i]));
			}
			for (int i = 0; i <= N_SQUARES; ++i) {
				double diff = rounew[i] - r[1][i];
				power_drou += diff * diff;
				r[1][i] = rounew[i];
			}
			if (power_drou < 1e-10 * N_SQUARES) {
				break;
			}
		}
		density[jj] = r[1][0];
		for (int i = 1; i <= N_SQUARES; ++i) {
			density[jj] += r[1][i];
		}
		density[jj] /= Ntotal_pores;
	}
	return density;
}
