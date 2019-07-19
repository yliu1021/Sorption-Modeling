#include <array>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <thread>

#include <ctime>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;

#include "constants.h"
#include "fast_dft_std.cpp"
#include "helpers.cpp"

#define WORKERS 1000

int main(int argc, char *argv[]) {
	setup_NL(); // for fast_dft

    int ITERS = 10;

    std::vector<std::array<double,N_SQUARES>> grids;

    if (argc == 2) {
        string grid_path(argv[1]);
        std::array<double, N_SQUARES> grid = load_grid(grid_path);
        for (int i = 0; i < WORKERS; ++i) {
            grids.push_back(grid);
        }

        std::array<double, N_ADSORP+1> linear = linear_curve();
        std::vector<double> costs;

        srand(time(NULL));

        for (int i = 0; i < ITERS; ++i) {
            // Calculate survival rates
            for (int j = 0; j < WORKERS; ++j) {
                // TODO: Lower mutation rate (currently, good grids evolve badly far too often)
                // TODO: After lowered mutation rate, avoid recalculating dft for every iteration
                toggle_random(grids[j]);
                costs.push_back(mean_abs_error(linear, run_dft(grids[j])));
                // cout << costs[j] << "\t";
            }
            if (i == (ITERS - 1)) { break; }
            cout << "MIN COST FOR ITER " << i << ": " << *std::min_element(costs.begin(), costs.end()) << endl;
            // if (*std::min_element(costs.begin(), costs.end()) < 0.01945) { break; } // 0.0193528 from 78, 
            std::vector<double> norm_costs(costs);
            normalizeArr(norm_costs.begin(), norm_costs.end(), *std::min_element(norm_costs.begin(), norm_costs.end()), *std::max_element(norm_costs.begin(), norm_costs.end()));

            // Reproduce suitable grids
            for (int j = 0; j < WORKERS; ++j) {
                 if (((double)rand()/(RAND_MAX)) < (1-norm_costs[j])) {
                    std::array<double,N_SQUARES> copy_grid(grids[j]);
                    grids.push_back(copy_grid);
                    // costs.push_back(costs[j]);
                    norm_costs.push_back(norm_costs[j]);
                }
            }

            // cout << "AFTER REPRODUCTION: " << grids.size() << " grids alive" << endl;
            // for (int j = 0; j < WORKERS; j++) { cout << costs[j] << "\t"; }

            // Kill unsuitable grids
            while (grids.size() > WORKERS) {
                int rand_grid = rand() % grids.size();
                if (((double)rand()/(RAND_MAX)) < norm_costs[rand_grid]) {
                    // cout << grids.size() << " " << norm_costs.size() << " " << costs.size() << endl;
                    grids.erase(grids.begin()+rand_grid);
                    norm_costs.erase(norm_costs.begin()+rand_grid);
                    // costs.erase(costs.begin()+rand_grid);
                }
            }

            // cout << "AFTER NATURAL SELECTION: " << grids.size() << " grids alive" << endl;

            costs.clear();

            // string grid_file = grid_path + "optimized.csv"

			// bool write_success = write_density(pred_density, density_file);
			// if (!write_success) return -1;
        }

        // Return best grid
        std::array<double,N_SQUARES> best_grid = grids[std::min_element(costs.begin(), costs.end()) - costs.begin()];
        for (int i = 0; i < N_SQUARES; ++i) {
            if (i % 20 == 0) { cout << endl; }
            cout << best_grid[i] << ",";
        }
        cout << "BEST COST: " << *std::min_element(costs.begin(), costs.end()) << endl;
        

    } else {
        cerr << "Invalid cmd line arguments" << endl;
    }

    return 0;
}