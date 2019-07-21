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

// TODO: Multithread

constexpr int WORKERS = 1000;
constexpr int ITERS = 10;
constexpr double MUT_RATE = 0.6; // Mutation rate

int main(int argc, char *argv[]) {

    if (argc == 2) {

        setup_NL(); // for fast_dft

        string grid_path(argv[1]);
        array<double, N_SQUARES> start_grid = load_grid(grid_path);
        array<double, N_ADSORP+1> lin_curve = linear_curve();
        double grid_cost = mean_abs_error(lin_curve, run_dft(start_grid));

        vector<array<double,N_SQUARES>> grids(WORKERS, start_grid);
        vector<double> costs(WORKERS, grid_cost);

        srand(time(NULL)); // Generate random seed for random number generation

        // double min_cost = *min_element(costs.begin(), costs.end());

        for (int i = 0; i < ITERS; ++i) {
            // Calculate survival rates
            for (int j = 0; j < WORKERS; ++j) {
                if (((double)rand()/(RAND_MAX)) < MUT_RATE) {
                    toggle_random(grids[j]);
                    costs[j] = mean_abs_error(lin_curve, run_dft(grids[j]));
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
            normalizeVec(norm_costs);

            // Kill unsuitable grids
            for (int j = WORKERS-1; j >= 0; --j) {
                if (((double)rand()/(RAND_MAX)) < norm_costs[j]) {
                    grids.erase(grids.begin()+j);
                    norm_costs.erase(norm_costs.begin()+j);
                    costs.erase(costs.begin()+j);
                }
            }

            cout << "After natural selection: " << grids.size() << " grids alive" << endl;

            // Reproduce suitable grids
            while (grids.size() < WORKERS) {
                int rand_grid = rand() % grids.size();
                if (((double)rand()/(RAND_MAX)) < (1-norm_costs[rand_grid])) {
                    array<double,N_SQUARES> copy_grid(grids[rand_grid]);
                    grids.push_back(copy_grid);
                    costs.push_back(costs[rand_grid]);
                    norm_costs.push_back(norm_costs[rand_grid]);
                }
            }

            cout << "After reproduction: " << grids.size() << " grids alive" << endl;

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