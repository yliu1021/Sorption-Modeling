#include <array>
#include <iostream>
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
#include "helpers.h"

// TODO: Kill grids before reproducing
// TODO: Multithread

constexpr int WORKERS = 200;
constexpr int ITERS = 10;
constexpr double MUT_RATE = 0.6; // Mutation rate

int main(int argc, char *argv[]) {
	setup_NL(); // for fast_dft

    vector<array<double,N_SQUARES>> grids;

    if (argc == 2) {
        string grid_path(argv[1]);
        array<double, N_SQUARES> grid = load_grid(grid_path);
        array<double, N_ADSORP+1> linear = linear_curve();
        vector<double> costs;
        double grid_cost = mean_abs_error(linear, run_dft(grid));
        for (int i = 0; i < WORKERS; ++i) {
            grids.push_back(grid);
            costs.push_back(grid_cost);
        }

        srand(time(NULL));

        // double min_cost = *min_element(costs.begin(), costs.end());

        for (int i = 0; i < ITERS; ++i) {
            // Calculate survival rates
            for (int j = 0; j < WORKERS; ++j) {
                if (((double)rand()/(RAND_MAX)) < MUT_RATE) {
                    toggle_random(grids[j]);
                    costs[j] = mean_abs_error(linear, run_dft(grids[j]));
                }
            }
            if (i == (ITERS - 1)) { break; }
            double min_cost_iter = *min_element(costs.begin(), costs.end());
            cout << "Minimum cost for iteration " << i << ": " << min_cost_iter << endl;
            // if (min_cost_iter < min_cost) {
            //     min_cost = min_cost_iter;
            //     cout << "FOUND NEW BEST GRID WITH COST: " << min_cost_iter;
            //     array<double,N_SQUARES> best_grid = grids[min_element(costs.begin(), costs.end()) - costs.begin()];
            //     for (int i = 0; i < N_SQUARES; ++i) {
            //         if (i % 20 == 0) { cout << endl; }
            //         cout << best_grid[i] << ",";
            //     }
            //     cout << endl;
            // }
            vector<double> norm_costs(costs);
            normalizeVec(costs);
            // normalizeArr(norm_costs.begin(), norm_costs.end(), *min_element(norm_costs.begin(), norm_costs.end()), *max_element(norm_costs.begin(), norm_costs.end()));

            // Reproduce suitable grids
            for (int j = 0; j < WORKERS; ++j) {
                 if (((double)rand()/(RAND_MAX)) < (1-norm_costs[j])) {
                    array<double,N_SQUARES> copy_grid(grids[j]);
                    grids.push_back(copy_grid);
                    costs.push_back(costs[j]);
                    norm_costs.push_back(norm_costs[j]);
                }
            }

            cout << "After reproduction: " << grids.size() << " grids alive" << endl;

            // Kill unsuitable grids
            while (grids.size() > WORKERS) {
                int rand_grid = rand() % grids.size();
                if (((double)rand()/(RAND_MAX)) < norm_costs[rand_grid]) {
                    grids.erase(grids.begin()+rand_grid);
                    norm_costs.erase(norm_costs.begin()+rand_grid);
                    costs.erase(costs.begin()+rand_grid);
                }
            }

            cout << "After natural selection: " << grids.size() << " grids alive" << endl;

            // string grid_file = grid_path + "optimized.csv"

			// bool write_success = write_density(pred_density, density_file);
			// if (!write_success) return -1;
        }

        // Return best grid
        array<double,N_SQUARES> best_grid = grids[min_element(costs.begin(), costs.end()) - costs.begin()];
        for (int i = 0; i < N_SQUARES; ++i) {
            if (i % 20 == 0) { cout << endl; }
            cout << best_grid[i] << ",";
        }
        cout << "BEST COST: " << *min_element(costs.begin(), costs.end()) << endl;
    } else {
        cerr << "Invalid cmd line arguments" << endl;
    }

    return 0;
}