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

constexpr int population_size = 1000;
constexpr int max_iters = 150;
constexpr double mutation_rate = 0.6; // Mutation rate

constexpr void (*mutate_function)(array<double, N_SQUARES>&) = &toggle_random;
constexpr double (*cost_function)(const array<double,N_ADSORP+1>&, const array<double,N_ITER+1>&) = &mean_abs_error;
array<double, N_ADSORP+1> target_curve = linear_curve(); 

// Whether to artificially boost the reproduction rate of beneficial mutations. 
// When set to true starting from high costs (> 0.1-0.2 MSE), this option 
// vastly increases the robustness of the solution at a large expense of speed.
constexpr bool boost_positive_muts = false;
constexpr int boost_factor = 0.05 * population_size * mutation_rate;

constexpr bool save_iters = true;
string save_path = "evol_iter_grids/1/";

// Returns the number of grids loaded
int load_grids(vector<array<double,N_SQUARES>> &grids, vector<double> &costs, const string &grid_dir) {
    // TODO: (Low priority) improve reading of files/dft function. Currently inefficient
    for (int i = 0; true; ++i) {
        array<double,N_SQUARES> grid;
        char grid_name[20];
        sprintf(grid_name, "grid_%04d.csv", i);
        string grid_file = grid_dir + grid_name;
        ifstream ifile(grid_file);
        if (ifile) {
            grid = load_grid(grid_file);
        } else {
            return i;
        }
        ifile.close();
        double grid_cost = (*cost_function)(target_curve, run_dft(grid));
        grids.push_back(grid);
        costs.push_back(grid_cost);
    }
}

void mutate_grids(vector<array<double,N_SQUARES>> &grids, vector<double> &costs) {
    // TODO: mutated grids should only be copies
    for (int i = 0, size = grids.size(); i < size; ++i) {
        if (((double)rand()/(RAND_MAX)) < mutation_rate) {
            double orig_cost = costs[i];
            (*mutate_function)(grids[i]);
            costs[i] = (*cost_function)(target_curve, run_dft(grids[i]));
            if (boost_positive_muts) {
                if (costs[i] < orig_cost) {
                    array<double,N_SQUARES> copy_grid(grids[i]);
                    for (int j = 0; j < boost_factor; ++j) {
                        grids.push_back(copy_grid);
                        costs.push_back(costs[i]);
                    }
                }
            }
        }
    }
}

void kill_grids(vector<array<double,N_SQUARES>> &grids, vector<double> &costs, vector<double> &norm_costs) {
    // Kill unsuitable grids
    // TODO: This needs to be improved now that artificial boosting of reproduction can create 100k+ grids. Perhaps
    // shuffle the grids and iterate through rather than generating two random numbers every iteration
    while (grids.size() >= population_size) {
        int rand_grid = rand() % grids.size();
        if (((double)rand()/(RAND_MAX)) < norm_costs[rand_grid]) {
            grids.erase(grids.begin()+rand_grid);
            norm_costs.erase(norm_costs.begin()+rand_grid);
            costs.erase(costs.begin()+rand_grid);
            if (grids.size()%1000 == 0) { cout << "Killing grids, " << grids.size() << " grids left. " << endl; }
            // if (grids.size()%5000 == 0) { standardizeVec(norm_costs); normalizeVec(norm_costs); }
        }
    }
    for (int i = grids.size()-1; i >= 0; --i) {
        if (((double)rand()/(RAND_MAX)) < norm_costs[i]) {
            grids.erase(grids.begin()+i);
            norm_costs.erase(norm_costs.begin()+i);
            costs.erase(costs.begin()+i);
        }
    }
}

void reproduce_grids(vector<array<double,N_SQUARES>> &grids, vector<double> &costs, vector<double> &norm_costs) {
    // Reproduce suitable grids
    while (grids.size() < population_size) {
        int rand_grid = rand() % grids.size();
        if (((double)rand()/(RAND_MAX)) < (1-norm_costs[rand_grid])) {
            array<double,N_SQUARES> copy_grid(grids[rand_grid]);
            grids.push_back(copy_grid);
            costs.push_back(costs[rand_grid]);
            norm_costs.push_back(norm_costs[rand_grid]);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc == 2) {
        setup_NL(); // for fast_dft
        srand(time(NULL)); // Generate random seed for random number generation

        vector<array<double,N_SQUARES>> grids;
        vector<double> costs;
		string grid_dir(argv[1]);
        if (grid_dir.back() != '/') { grid_dir = grid_dir + "/"; }
        if (int num_start = load_grids(grids, costs, grid_dir)) {
            cout << "Loaded " << num_start << " grids. Starting evolution." << endl;
        } else {
            cerr << "ERROR: Failed to load grids." << endl;
            return 1;
        }
        double min_cost = *min_element(costs.begin(), costs.end());

        for (int i = 0; i < max_iters; ++i) {
            mutate_grids(grids, costs);
            cout << "After mutations: " << grids.size() << " grids alive" << endl;

            vector<double>::iterator min_cost_iter = min_element(costs.begin(), costs.end());
            cout << "Minimum cost for iteration " << i << ": " << *min_cost_iter << endl;
            if (*min_cost_iter < min_cost) {
                min_cost = *min_cost_iter;
                cout << "FOUND NEW BEST GRID WITH COST: " << *min_cost_iter;
                array<double,N_SQUARES> best_grid = grids[min_element(costs.begin(), costs.end()) - costs.begin()];
                write_grid(best_grid, cout);
                cout << endl;
            }

            if (save_iters) {
                char grid_name[20];
                sprintf(grid_name, "grid_%04d.csv", i);
                char density_name[20];
                sprintf(density_name, "density_%04d.csv", i);
                string grid_file = save_path + grid_name;
                string density_file = save_path + density_name;

                array<double, N_SQUARES> best_grid = grids[min_cost_iter - costs.begin()];
                array<double, N_ITER + 1> pred_density = run_dft(best_grid);

                if (!write_grid(best_grid, grid_file)) { return 1; }
                if (!write_density(pred_density, density_file)) { return 1; }
            }

            vector<double> norm_costs(costs);
            standardizeVec(norm_costs);
            normalizeVec(norm_costs);

            kill_grids(grids, costs, norm_costs);
            cout << "After natural selection: " << grids.size() << " grids alive" << endl;

            reproduce_grids(grids, costs, norm_costs);
            cout << "After reproduction: " << grids.size() << " grids alive" << endl;
        }
        cout << endl << "================================================" << endl;
        cout << "BEST COST AT FINAL ITER: " << *min_element(costs.begin(), costs.end()) << endl;
        cout << "BEST COST THROUGH ALL ITERS: " << min_cost << endl;
    } else {
        cerr << "Invalid cmd line arguments" << endl;
    }
    return 0;
}