#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>


using namespace std;
/*
fast_dft usage:
	-g --grids: The directory containing all the grid_####.csv files
	-r --results: The directory to store all the density_####.csv files
	fast_dft -g path/to/grids -r path/to/results
 */

#define GRID_SIZE 20
#define N_SQUARES (GRID_SIZE * GRID_SIZE)
#define N_ITER 80
#define N_ADSORP 40
#define STEP_SIZE 0.025

#define KB 0.0019872041
#define T 298.0
#define Y 1.5
#define TC 647.0
const double BETA = 1 / (KB * T);
const double MUSAT = -2.0 * KB * TC;
const double C = 4.0;
const double WFF = -2.0 * MUSAT / C;

/*
Computes and returns the density of a grid
	grid: a N_SQUARES long array of doubles
	The returned pointer to an array must be freed via:
		delete[] density
 */
double *run_dft(double grid[]) {
	double r[4][N_SQUARES + 1];
	
	double Lx = 0.0;
	for (int i = 1; i <= N_SQUARES; ++i) {
		r[0][i] = Lx;
		Lx += 1;
		if (i % GRID_SIZE == 0)
			Lx = 0;
	}
	double Ly = 0.0;
	for (int i = 1; i <= N_SQUARES; ++i) {
		r[1][i] = Ly;
		if (i % GRID_SIZE == 0)
			Ly += 1;
	}
	
	for (int i = 1; i <= N_SQUARES; ++i) {
		r[2][i] = grid[i - 1];
	}
	
	double Ntotal_pores = 0.0;
	for (int i = 0; i <= N_SQUARES; ++i) {
		Ntotal_pores += r[2][i];
	}
	
	double rc = 1.01;
	double rc_square = rc * rc;
	int NN[N_SQUARES + 1];
	int NL[N_SQUARES + 1][N_SQUARES];
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
	
	for (int i = 1; i <= N_SQUARES; ++i) {
		r[3][i] = r[2][i];
	}
	
	double *density = new double[N_ITER + 1];
	for (int jj = 0; jj <= N_ITER; ++jj) {
		double RH;
		double muu;
		if (jj <= N_ADSORP) {
			RH = jj * STEP_SIZE;
		} else {
			RH = N_ADSORP*STEP_SIZE - (jj - N_ADSORP)*STEP_SIZE;
		}
		if (RH == 0.0) {
			muu = -90;
		} else {
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
				vi[i] = WFF * (r[3][a1] + Y * (1 - r[2][a1])) +
						WFF * (r[3][a2] + Y * (1 - r[2][a2])) +
						WFF * (r[3][a3] + Y * (1 - r[2][a3])) +
						WFF * (r[3][a4] + Y * (1 - r[2][a4]));
				vi[i] += muu;
			}
			// rounew = rou(vi,r)
			double rounew[N_SQUARES + 1];
			for (int i = 1; i <= N_SQUARES; ++i) {
				rounew[i] = r[2][i] / (1 + exp(-BETA * vi[i]));
			}
			
			double power_drou = 0.0;
			for (int i = 0; i <= N_SQUARES; ++i) {
				double diff = rounew[i] - r[3][i];
				power_drou += diff * diff;
			}
			power_drou /= N_SQUARES;
			for (int i = 0; i <= N_SQUARES; ++i) {
				r[3][i] = rounew[i];
			}
			if (power_drou < 1e-10) {
				break;
			}
			if (c == 100000000) {
				cout << "error" << endl;
			}
		}
		density[jj] = 0.0;
		for (int i = 0; i <= N_SQUARES; ++i) {
			density[jj] += r[3][i];
		}
		density[jj] /= Ntotal_pores;
	}
	return density;
}

/*
Loads a grid from a grid_####.csv file
The returned array to a grid array must be freed via:
	delete[] grid
 */
double *load_grid(const string &path) {
    ifstream grid_file;
    grid_file.open(path);
	if (!grid_file.is_open()) {
		cerr << "File " << path << " doesn't exist" << endl;
		return nullptr;
	}
	
	double *grid = new double[N_SQUARES];
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
			return nullptr;
		}
	}
	
	grid_file.close();
	return grid;
}

double *load_density(const string &path) {
	ifstream density_file;
	density_file.open(path);
	if (!density_file.is_open()) {
		cerr << "File " << path << " doesn't exist" << endl;
		return nullptr;
	}
	
	double *densities = new double[N_ITER + 1];
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

bool write_density(const double *density, const string &path) {
	ofstream density_file;
	density_file.open(path);
	if (!density_file.is_open()) {
		cerr << "Could not open/create file" << endl;
		return false;
	}
	
	density_file << ",0" << endl;
	for (int i = 0; i <= N_ITER; ++i) {
		density_file << i << "," << setprecision(17) << density[i] << endl;;
	}
	
	density_file.close();
	return true;
}

int main(int argc, char *argv[]) {
	string base_dir(argv[1]);
	if (base_dir.back() != '/')
		base_dir = base_dir + "/";
	// base_dir = "./generative_model/step##/"
	string grid_dir = base_dir + "grids/";
	string density_dir = base_dir + "results/";
	for (int i = 0; i < 300; ++i) {
		char grid_name[20];
		sprintf(grid_name, "grid_%04d.csv", i);
		char density_name[20];
		sprintf(density_name, "density_%04d.csv", i);
		
		string grid_file = grid_dir + grid_name;
		string density_file = density_dir + density_name;
		
		double *grid = load_grid(grid_file);
		if (grid == nullptr) return -1;
		double *pred_density = run_dft(grid);
		if (pred_density == nullptr) return -1;
		if (!write_density(pred_density, density_file)) return -1;

		delete[] grid;
		delete[] pred_density;
	}
	return 0;
}