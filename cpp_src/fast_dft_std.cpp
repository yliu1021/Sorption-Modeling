#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>
#include <string>
#include <thread>

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "constants.h"

using namespace std;
/*
fast_dft usage:
	fast_dft ...folder/containing/grid/folder/
*/

#define KB 0.0019872041
#define T 298.0
#define Y 1.5
#define TC 647.0
const double BETA = 1 / (KB * T);
const double MUSAT = -2.0 * KB * TC;
const double C = 4.0;
const double WFF = -2.0 * MUSAT / C;

int NL[N_SQUARES + 1][N_SQUARES];

// setup code for run_dft
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
}

/*
Computes and returns the density of a grid
	grid: a N_SQUARES long array of doubles
	The returned pointer to an array must be freed via:
		delete[] density
 */
std::array<double,N_ITER+1> run_dft(std::array<double,N_SQUARES> grid) {
    std::array<double,N_ITER+1> density;
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
		density[jj] = 0.0;
		for (int i = 0; i <= N_SQUARES; ++i) {
			density[jj] += r[1][i];
		}
		density[jj] /= Ntotal_pores;
	}
	return density;
}

/*
Loads a grid from a grid_####.csv file
 */
std::array<double,N_SQUARES> load_grid(istream &grid_file) {
    std::array<double,N_SQUARES> grid;
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

std::array<double,N_SQUARES> load_grid(const string &path) {
    ifstream grid_file;
    grid_file.open(path);
	if (!grid_file.is_open()) {
		cerr << "File " << path << " doesn't exist" << endl;
        exit(1);
	}
	
	std::array<double,N_SQUARES> grid = load_grid(grid_file);
	
	grid_file.close();
	return grid;
}

std::array<double,N_ITER+1> load_density(const string &path) {
	ifstream density_file;
	density_file.open(path);
	if (!density_file.is_open()) {
		cerr << "File " << path << " doesn't exist" << endl;
        exit(1);
	}
	
    std::array<double,N_ITER+1> densities;
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

void write_density(std::array<double,N_ITER+1> density, ostream &density_file) {
	density_file << ",0" << endl;
	for (int i = 0; i <= N_ITER; ++i) {
		density_file << i << "," << setprecision(17) << density[i] << endl;;
	}
}

bool write_density(std::array<double,N_ITER+1> density, const string &path) {
	ofstream density_file;
	density_file.open(path);
	if (!density_file.is_open()) {
		cerr << "Could not open/create file" << endl;
		return false;
	}
	
	write_density(density, density_file);
	
	density_file.close();
	return true;
}