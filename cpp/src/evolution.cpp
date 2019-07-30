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
// TODO: Duplicate AND mutate
// TODO: Read from folder

constexpr int WORKERS = 2500;
constexpr int ITERS = 150;
constexpr double MUT_RATE = 0.6; // Mutation rate

// Whether to artificially boost the reproduction rate of beneficial mutations. 
// When set to true starting from high costs (> 0.1-0.2 MSE), this option 
// vastly increases the robustness of the solution at a large expense of speed.
constexpr bool BOOST_POSITIVE_MUTS = true;
constexpr int BOOST_FACTOR = 0.05 * WORKERS * MUT_RATE;

constexpr bool WRITE_OUTPUT = true;
string WRITE_FOLDER = "evol_iter_grids/1/";

int main(int argc, char *argv[]) {
    if (argc == 2) {
        setup_NL(); // for fast_dft

        // vector<double> step_size{0.3, 0.6, 1};
        // vector<double> step_height{0.3, 0.6, 1}
        // array<double, N_ADSORP+1> target_curve = step_function(step_height, step_size);

        // array<double, N_ADSORP+1> target_curve = heaviside_step_function(0.75);

        array<double, N_ADSORP+1> target_curve = circular_curve(1, true);
        // for (int i = 0; i < N_ADSORP+1; i++) {
        //     cout << target_curve[i] << endl;
        // }

        string grid_path(argv[1]);
        array<double, N_SQUARES> start_grid = load_grid(grid_path);

        double grid_cost = mean_abs_error(target_curve, run_dft(start_grid));

        vector<array<double,N_SQUARES>> grids(WORKERS, start_grid);
        vector<double> costs(WORKERS, grid_cost);

        srand(time(NULL)); // Generate random seed for random number generation

        double min_cost = *min_element(costs.begin(), costs.end());

        for (int i = 0; i < ITERS; ++i) {
            // Calculate survival rates
            for (int j = 0; j < WORKERS; ++j) {
                if (((double)rand()/(RAND_MAX)) < MUT_RATE) {
                    double orig_cost = costs[j];
                    toggle_random(grids[j]);
                    costs[j] = mean_abs_error(target_curve, run_dft(grids[j]));
                    if (BOOST_POSITIVE_MUTS) {
                        if (costs[j] < orig_cost) {
                            array<double,N_SQUARES> copy_grid(grids[j]);
                            for (int k = 0; k < BOOST_FACTOR; ++k) {
                                grids.push_back(copy_grid);
                                costs.push_back(costs[j]);
                            }
                        }
                    }
                }
            }

            double min_cost_iter = *min_element(costs.begin(), costs.end());
            cout << "Minimum cost for iteration " << i << ": " << min_cost_iter << endl;
            if (min_cost_iter < min_cost) {
                min_cost = min_cost_iter;
                cout << "FOUND NEW BEST GRID WITH COST: " << min_cost_iter;
                array<double,N_SQUARES> best_grid = grids[min_element(costs.begin(), costs.end()) - costs.begin()];
                for (int i = 0; i < N_SQUARES; ++i) {
                    if (i % 20 == 0) { cout << endl; }
                    cout << best_grid[i] << ",";
                }
                cout << endl;
            }

            cout << "After artificial reproduction: " << grids.size() << " grids alive" << endl;

            if (WRITE_OUTPUT) {
                char grid_name[20];
                sprintf(grid_name, "grid_%04d.csv", i);
                char density_name[20];
                sprintf(density_name, "density_%04d.csv", i);

                string grid_file = WRITE_FOLDER + grid_name;
                string density_file = WRITE_FOLDER + density_name;

                array<double, N_SQUARES> best_grid = grids[min_element(costs.begin(), costs.end()) - costs.begin()];
                array<double, N_ITER + 1> pred_density = run_dft(best_grid);

                if (!write_density(pred_density, density_file)) { return -1; }
                if (!write_grid(best_grid, grid_file)) { return -1; }
            }

            vector<double> norm_costs(costs);
            standardizeVec(norm_costs);
            normalizeVec(norm_costs);

            // cout << endl;
            // vector<double> debug(norm_costs);
            // sort(debug.begin(), debug.end());
            // for (int j = 0; j < debug.size(); j++) {
            //     cout << debug[j] << "\t";
            // }
            // cout << endl;

            // Kill unsuitable grids
            // TODO: This needs to be improved now that artificial boosting of reproduction can create 100k+ grids. Perhaps
            // shuffle the grids and iterate through rather than generating a random number every iteration
            while (grids.size() > WORKERS) {
                int rand_grid = rand() % grids.size();
                if (((double)rand()/(RAND_MAX)) < norm_costs[rand_grid]) {
                    grids.erase(grids.begin()+rand_grid);
                    norm_costs.erase(norm_costs.begin()+rand_grid);
                    costs.erase(costs.begin()+rand_grid);
                    if (grids.size()%1000 == 0) { cout << "Killing grids, " << grids.size() << " grids left. " << endl; }
                    // if (grids.size()%5000 == 0) { standardizeVec(norm_costs); normalizeVec(norm_costs); }
                }
            }
            for (int j = grids.size()-1; j >= 0; --j) {
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

        // // Return best grid
        // array<double,N_SQUARES> best_grid = grids[min_element(costs.begin(), costs.end()) - costs.begin()];
        // for (int i = 0; i < N_SQUARES; ++i) {
        //     if (i % 20 == 0) { cout << endl; }
        //     cout << best_grid[i] << ",";
        // }
        cout << endl << "================================================" << endl;
        cout << "BEST COST AT FINAL ITER: " << *min_element(costs.begin(), costs.end()) << endl;
        cout << "BEST COST THROUGH ALL ITERS: " << min_cost << endl;
    } else {
        cerr << "Invalid cmd line arguments" << endl;
    }

    return 0;
}