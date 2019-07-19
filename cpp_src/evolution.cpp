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

#define WORKERS 100

int main(int argc, char *argv[]) {
	setup_NL(); // for fast_dft

    int ITERS = 50;

    std::vector<std::array<double,N_SQUARES>> grids;

    if (argc == 2) {
        string grid_path(argv[1]);
        std::array<double, N_SQUARES> grid = load_grid(grid_path);
        for (int i = 0; i < WORKERS; ++i) {
            grids.push_back(grid);
        }

        std::array<double, N_ADSORP+1> linear = linear_curve();
        std::array<double, WORKERS> costs;

        srand(time(NULL));

        for (int i = 0; i < ITERS; ++i) {
            // Calculate survival rates
            for (int j = 0; j < WORKERS; ++j) {
                toggle_random(grids[j]);
                costs[j] = mean_abs_error(linear, run_dft(grids[j]));
                // cout << costs[j] << "\t";
            }
            cout << "MIN COST FOR THIS ITER: " << *std::min_element(costs.begin(), costs.end()) << endl;
            normalizeArr(costs.begin(), costs.end(), *std::min_element(costs.begin(), costs.end()), *std::max_element(costs.begin(), costs.end()));

            // Reproduce suitable grids
            for (int j = 0; j < WORKERS; ++j) {
                 if (((double)rand()/(RAND_MAX)) < (1-costs[j])) {
                    std::array<double,N_SQUARES> copy = grids[j];
                    grids.push_back(copy);
                }
            }

            // cout << "AFTER REPRODUCION: " << grids.size() << endl;

            // Kill unsuitable grids
            while (grids.size() > WORKERS) {
                int rand_grid = rand() % grids.size();
                if (((double)rand()/(RAND_MAX)) < costs[rand_grid]) {
                    grids.erase(grids.begin()+rand_grid);
                }
            }
            // cout << "AFTER NATURAL SELECTION: " << grids.size() << endl;
            // for (int j = WORKERS-1; j >= 0; j--) {
            //     cout << costs[j] << "\t";
            //     if (((double)rand()/(RAND_MAX)) < costs[j]) {
            //         grids.erase(grids.begin()+j);
            //     }
            // }

            // // Reproduce suitable grids
            // while (grids.size() < WORKERS) {
            //     int rand_grid = rand() % grids.size();
            //     if (((double)rand()/(RAND_MAX)) < (1-costs[rand_grid])) {
            //         std::array<double,N_SQUARES> copy = grids[rand_grid];
            //         grids.push_back(copy);
            //     }
            // }

            // Return best grid
        }
    } else {
        cerr << "Invalid cmd line arguments" << endl;
    }

    return 0;
}