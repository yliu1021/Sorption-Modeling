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

    int ITERS = 300;

    if (argc == 2) {
        array<double,N_SQUARES> target_grid;
        for (int i = 0; i < N_SQUARES; i++) { target_grid[i] = 1; }
        // for (int i = 0; i < 20; i++) {
        //     for (int j = 0; j < 20; j++) {
        //         if (i == 0 || j == 0 || i == 19 || j == 19) {
        //             target_grid[20*i+j] = 0;
        //         } else {
        //             target_grid[20*i+j] = 1;
        //         }
        //     }
        // }
        array<double,N_ITER+1> dft_den = run_dft(target_grid);
        array<double, N_ADSORP+1> target_curve;
        for (int i = 0; i < N_ADSORP+1; i++) {
            target_curve[i] = dft_den[i];
        }

        // array<double, N_ADSORP+1> target_curve = linear_curve();

        string grid_path(argv[1]);
        // array<double, N_SQUARES> start_grid = load_grid(grid_path);
        srand(time(NULL)); // Generate random seed for random number generation
        array<double, N_SQUARES> start_grid = random_grid();

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

            char grid_name[20];
            sprintf(grid_name, "grid_%04d.csv", i);
            char density_name[20];
            sprintf(density_name, "density_%04d.csv", i);
            string save_folder = "./evol_iter_grids/empty_steepest/";
            string grid_file = save_folder + grid_name;
            string density_file = save_folder + density_name;

            array<double, N_SQUARES> best_grid = grids[min_cost_it - costs.begin()];
            array<double, N_ITER + 1> pred_density = run_dft(best_grid);

            if (!write_grid(best_grid, grid_file)) { return 1; }
            if (!write_density(pred_density, density_file)) { return 1; }

            if (*min_cost_it+0.000000000000001 > last_cost) {
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
