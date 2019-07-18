#include <array>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <thread>

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;

#include "constants.h"
#include "fast_dft.cpp"
#include "helpers.cpp"

int main(int argc, char *argv[]) {
	setup_NL(); // for fast_dft

    int ITERS = 5;

    if (argc == 2) {
        string grid_path(argv[1]);
        double *grid1 = load_grid(grid_path);
        if (grid1 == nullptr) { return -1; }

        std::array<double, N_SQUARES> grid;
        for (int i = 0; i < N_SQUARES; i++) {
            grid[i] = grid1[i];
        }

        for (int i = 0; i < ITERS; ++i) {
			double *pointer_density = run_dft(grid1);
            if (pointer_density == nullptr) { return -1; }
            std::array<double, N_ADSORP+1> pred_density;
            for (int i = 0; i < N_ADSORP+1; i++) {
                pred_density[i] = pointer_density[i];
            }
            delete[] pointer_density;

            std::array<double, N_ADSORP+1> linear = linear_curve();
            // kullback_leibler_divergence(linear, pred_density);
            // cout << "COST: " << kullback_leibler_divergence(linear, pred_density) << endl;
            cout << "COST: " << mean_abs_error(linear, pred_density) << endl;
            toggle_random(grid);

            for (int i = 0; i < N_SQUARES; i++) {
                grid1[i] = grid[i];
            }
        }

        delete[] grid1;
    } else {
        cerr << "Invalid cmd line arguments" << endl;
    }

	// if (argc == 1) {
	// 	double *grid = load_grid(cin);
	// 	double *density = run_dft(grid);
		
	// 	write_density(density, cout);
		
	// 	delete[] grid;
	// 	delete[] density;
	// } else if (argc == 2) {
	// 	string base_dir(argv[1]);
	// 	if (base_dir.back() != '/')
	// 		base_dir = base_dir + "/";
	// 	// base_dir = "./generative_model/step##/"
	// 	string grid_dir = base_dir + "grids/";
	// 	string density_dir = base_dir + "results/";
	// 	string cmd = "mkdir -p " + density_dir;
	// 	system(cmd.c_str());

	// 	for (int i = 0; true; ++i) {
	// 		char grid_name[20];
	// 		sprintf(grid_name, "grid_%04d.csv", i);
	// 		char density_name[20];
	// 		sprintf(density_name, "density_%04d.csv", i);
		
	// 		string grid_file = grid_dir + grid_name;
	// 		string density_file = density_dir + density_name;
		
	// 		double *grid = load_grid(grid_file);
	// 		if (grid == nullptr) break;
	// 		double *pred_density = run_dft(grid);
	// 		if (pred_density == nullptr) return -1;
			
	// 		bool write_success = write_density(pred_density, density_file);
	// 		if (!write_success) return -1;

	// 		delete[] grid;
	// 		delete[] pred_density;
	// 	}
	// } else {
	// 	cerr << "Invalid cmd line arguments" << endl;
	// }
	return 0;
}