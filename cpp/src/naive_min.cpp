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

int main(int argc, char *argv[]) {
	setup_NL(); // for fast_dft

    int ITERS = 2;

    if (argc == 2) {
        array<double, N_ADSORP+1> target_curve = heaviside_step_function();

        string grid_path(argv[1]);
        array<double, N_SQUARES> start_grid = load_grid(grid_path);

        array<array<double, N_SQUARES>, N_SQUARES> grids;
        grids.fill(start_grid);
        array<double, N_SQUARES> costs;

        double last_cost = 10; // Initialize to arbitrary large number
        array<double, N_SQUARES> last_grid = start_grid;

        for (int i = 0; i < ITERS; ++i) {
            for (int j = 0; j < N_SQUARES; j++) {
                grids[j][j] = 1-grids[j][j];
                costs[j] = mean_abs_error(target_curve, run_dft(grids[j]));
            }
            double* min_cost_it = min_element(costs.begin(), costs.end());
            cout << "Minimum cost for iteration " << i << ": " << *min_cost_it << endl;
            if (*min_cost_it > last_cost) {
                cout << "No single cell improvements left" << endl;
                break;
            }
            last_cost = *min_cost_it;
            last_grid = grids[min_cost_it-costs.begin()];
            grids.fill(grids[min_cost_it-costs.begin()]);
        }
        cout << "================================================" << endl;
        cout << "Final grid: " << endl;
        for (int i = 0; i < N_SQUARES; ++i) {
            if (i % 20 == 0) { cout << endl << last_grid[i]; }
            else { cout << "," << last_grid[i]; }
        }
        cout << endl << endl;
    } else {
        cerr << "Invalid cmd line arguments" << endl;
    }
    return 0;
}