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
#include "helpers.h"

#define WORKERS 200
#define MUT_RATE 0.6 // Mutation rate
#define ITERS 10

int main(int argc, char *argv[]) {
	setup_NL(); // for fast_dft

    std::vector<std::array<double,N_SQUARES>> grids;

    if (argc == 2) {
        string grid_path(argv[1]);
        std::array<double, N_SQUARES> grid = load_grid(grid_path);
        std::array<double, N_ADSORP+1> linear = linear_curve();
        std::vector<double> costs;
        double grid_cost = mean_abs_error(linear, run_dft(grid));
        for (int i = 0; i < WORKERS; ++i) {
            grids.push_back(grid);
            costs.push_back(grid_cost);
        }

        srand(time(NULL));

        double min_cost = *std::min_element(costs.begin(), costs.end());

        for (int i = 0; i < ITERS; ++i) {
            // Calculate survival rates
            for (int j = 0; j < WORKERS; ++j) {
                if (((double)rand()/(RAND_MAX)) < MUT_RATE) {
                    toggle_random(grids[j]);
                    costs[j] = mean_abs_error(linear, run_dft(grids[j]));
                }
                // cout << costs[j] << "\t";
            }
            if (i == (ITERS - 1)) { break; }
            double min_cost_iter = *std::min_element(costs.begin(), costs.end());
            cout << "Minimum cost for iteration " << i << ": " << min_cost_iter << endl;
            // if (*std::min_element(costs.begin(), costs.end()) < 0.01945) { break; } // 0.0193528 from 78, 
            if (min_cost_iter < min_cost) {
                min_cost = min_cost_iter;
                cout << "FOUND NEW BEST GRID WITH COST: " << min_cost_iter;
                std::array<double,N_SQUARES> best_grid = grids[std::min_element(costs.begin(), costs.end()) - costs.begin()];
                for (int i = 0; i < N_SQUARES; ++i) {
                    if (i % 20 == 0) { cout << endl; }
                    cout << best_grid[i] << ",";
                }
                cout << endl;
            }
            std::vector<double> norm_costs(costs);
            normalizeArr(norm_costs.begin(), norm_costs.end(), *std::min_element(norm_costs.begin(), norm_costs.end()), *std::max_element(norm_costs.begin(), norm_costs.end()));

            // Reproduce suitable grids
            for (int j = 0; j < WORKERS; ++j) {
                 if (((double)rand()/(RAND_MAX)) < (1-norm_costs[j])) {
                    std::array<double,N_SQUARES> copy_grid(grids[j]);
                    grids.push_back(copy_grid);
                    costs.push_back(costs[j]);
                    norm_costs.push_back(norm_costs[j]);
                }
            }

            cout << "After reproduction: " << grids.size() << " grids alive" << endl;
            // for (int j = 0; j < WORKERS; j++) { cout << costs[j] << "\t"; }

            // Kill unsuitable grids
            int kill_iter = 0;
            while (grids.size() > WORKERS) {
                kill_iter++;
                if (kill_iter > 1000000000) { 
                    // Near convergence, all grids may have the same cost
                    break;
                }
                int rand_grid = rand() % grids.size();
                if (((double)rand()/(RAND_MAX)) < norm_costs[rand_grid]) {
                    // cout << grids.size() << " " << norm_costs.size() << " " << costs.size() << endl;
                    grids.erase(grids.begin()+rand_grid);
                    norm_costs.erase(norm_costs.begin()+rand_grid);
                    costs.erase(costs.begin()+rand_grid);
                }
            }

            if (kill_iter > 1000000000) {
                // Near convergence, all grids may have the same cost
                break;
            }

            cout << "After natural selection: " << grids.size() << " grids alive" << endl;

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