#include <array>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>

#include <ctime>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;

#include "constants.h"
#include "helpers.h"

int main(int argc, char *argv[]) {
	setup_NL(); // for fast_dft

    int MAX_ITERS = 300;

    string path = "../data_generation/results/density_0000.csv";
    array<double,N_ADSORP+1> target_curve = load_density(path);
    // for (int i = 0; i < N_ADSORP+1; ++i) {
    //     cout << target_curve[i] << endl;
    // }

    array<Grid, N_SQUARES> toggled_grids;
    toggled_grids.fill(random_grid());
    Grid best_grid = toggled_grids[0];
    array<double, N_SQUARES> toggled_costs;
    double min_cost = 100;

for (int iterations = 0; iterations < MAX_ITERS; iterations++) {
    for (unsigned i = 0; i < toggled_grids.size(); ++i) {
        toggled_grids[i][i] = 1 - toggled_grids[i][i];
        toggled_costs[i] = mean_abs_error(target_curve, run_dft(toggled_grids[i]));
    }
            
    double* min_cost_it = min_element(toggled_costs.begin(), toggled_costs.end());
            
    cout << "Minimum cost for iteration " << iterations << ": " << *min_cost_it << endl;

//            char grid_name[20];
//            sprintf(grid_name, "grid_%04d.csv", i);
//            char density_name[20];
//            sprintf(density_name, "density_%04d.csv", i);
//            string save_folder = "./evol_iter_grids/2/";
//            string grid_file = save_folder + grid_name;
//            string density_file = save_folder + density_name;
//            array<double, N_SQUARES> best_grid_iter = toggled_grids[min_cost_it - costs.begin()];
//            array<double, N_ITER+1> pred_density = run_dft(best_grid);
//            if (!write_grid(best_grid, grid_file)) { return 1; }
//            if (!write_density(pred_density, density_file)) { return 1; }

        if (*min_cost_it > min_cost) {
            cout << "No single cell improvements left" << endl;
            break;
        }
        min_cost = *min_cost_it;
        best_grid = toggled_grids[min_cost_it-toggled_costs.begin()];
        toggled_grids.fill(best_grid);
    }
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 19; ++j) {
            cout << best_grid[i*20+j] << ",";
        }
        cout << best_grid[i*20+19] << endl;
    }

    return 0;
}
