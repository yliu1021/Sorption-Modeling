#include <array>
#include <iostream>
// #include <fstream>
// #include <vector>
#include <string>
#include <math.h>
// #include <thread>

// #include <ctime>
// #include <cstring>
// #include <cmath>
// #include <cstdio>
// #include <cstdlib>

using namespace std;

#include "constants.h"
#include "helpers.h"

constexpr int BEES = 25;
constexpr int ITERS = 2000;
string save_path = "./swarm_grids/";

double sigmoid(double x) {
    return (1 / (1 + exp(-x)));
}

double sigmoid_inverse(double x) {
    return log(x / (1-x));
}

int main(int argc, char *argv[]) {
	setup_NL(); // for fast_dft
    srand(time(NULL)); // Generate random seed for random number generation

    array<double, N_ADSORP+1> target_curve = heaviside_step_function();

    // array<double, N_ADSORP+1> target_curve;
    // string path = "./1solid_den.csv";
    // array<double,N_ITER+1> whole_curve = load_density(path);
    // for (int i = 0; i < N_ADSORP+1; ++i) {
    //     target_curve[i] = whole_curve[i];
    // }

    array<array<double, N_SQUARES>, BEES> positions;
    array<array<double, N_SQUARES>, BEES> velocities;
    array<double, BEES> costs;

    array<array<double, N_SQUARES>, BEES> personal_best_grids;
    array<double,N_SQUARES> global_best_grid;
    array<double, BEES> personal_min_cost;
    personal_min_cost.fill(10.0);
    double global_min_cost = 10.0;

    double v_min = sigmoid_inverse(0.0025);
    double v_max = sigmoid_inverse(0.9975);

    // Initialize random positions and velocities
    for (int i = 0; i < BEES; ++i) {
        for (int j = 0; j < N_SQUARES; j++) {
            if (((double)rand()/(RAND_MAX)) < 0.5) {
                positions[i][j] = 0;
            } else {
                positions[i][j] = 1;
            }

            double v = ((double)rand()/(RAND_MAX)) * 0.995 + 0.0025;
            velocities[i][j] = sigmoid_inverse(v);
        }
    }

    for (int iteration = 0; iteration < ITERS; ++iteration) {

        double vsum = 0;
        // Update velocities
        double r1 = ((double)rand()/(RAND_MAX));
        double r2 = ((double)rand()/(RAND_MAX));
        for (int i = 0; i < BEES; ++i) {
            for (int j = 0; j < N_SQUARES; ++j) {
                double p_contrib = 0.5 * r1 * (personal_best_grids[i][j] - positions[i][j]);
                double g_contrib = 0.5 * r2 * (global_best_grid[j] - positions[i][j]);
                velocities[i][j] += (p_contrib + g_contrib);

                // cout << p_contrib << "\t\t\t" << g_contrib << endl;

                // Clip velocities
                velocities[i][j] = max(velocities[i][j], v_min);
                velocities[i][j] = min(velocities[i][j], v_max);
                vsum += abs(velocities[i][j]);
            }
        }
        cout << "AVERAGE VELOCITY: " << vsum/(BEES*N_SQUARES) << endl;

        // Update positions
        for (int i = 0; i < BEES; ++i) {
            for (int j = 0; j < N_SQUARES; ++j) {
                if (((double)rand()/(RAND_MAX)) < sigmoid(velocities[i][j])) {
                    // if (positions[i][j] != 0) { cout << "!"; }
                    positions[i][j] = 1;
                } else {
                    // if (positions[i][j] != 1) { cout << "!"; }
                    positions[i][j] = 0;
                }
            }
        }

        if (iteration % 5 == 0) {
            char grid_name[20];
            sprintf(grid_name, "grid_%04d.csv", iteration/5);
            char density_name[20];
            sprintf(density_name, "density_%04d.csv", iteration/5);
            string grid_file = save_path + grid_name;
            string density_file = save_path + density_name;

            array<double, N_ITER + 1> pred_density = run_dft(global_best_grid);

            if (!write_grid(global_best_grid, grid_file)) { return 1; }
            if (!write_density(pred_density, density_file)) { return 1; }
        }

        // Update costs
        for (int i = 0; i < BEES; ++i) {
            costs[i] = mean_abs_error(target_curve, run_dft(positions[i]));
            if (costs[i] < personal_min_cost[i]) {
                personal_min_cost[i] = costs[i];
                personal_best_grids[i] = positions[i];
            }
            if (costs[i] < global_min_cost) {
                global_min_cost = costs[i];
                global_best_grid = positions[i];
            }
        }
        cout << "Minimum cost at iteration " << iteration << ": " << global_min_cost << endl;

    }

    // array<array<double, N_SQUARES>, N_SQUARES> grids;
    // grids.fill(start_grid);
    // array<double, N_SQUARES> costs;

    // double last_cost = 10; // Initialize to arbitrary large number
    // array<double, N_SQUARES> last_grid = start_grid;

    // for (int i = 0; i < ITERS; ++i) {
    //     for (int j = 0; j < N_SQUARES; j++) {
    //         grids[j][j] = 1-grids[j][j];
    //         costs[j] = mean_abs_error(target_curve, run_dft(grids[j]));
    //     }
    //     double* min_cost_it = min_element(costs.begin(), costs.end());
    //     cout << "Minimum cost for iteration " << i << ": " << *min_cost_it << endl;

    //     char grid_name[20];
    //     sprintf(grid_name, "grid_%04d.csv", i);
    //     char density_name[20];
    //     sprintf(density_name, "density_%04d.csv", i);
    //     string save_folder = "./evol_iter_grids/2/";
    //     string grid_file = save_folder + grid_name;
    //     string density_file = save_folder + density_name;

    //     array<double, N_SQUARES> best_grid = grids[min_cost_it - costs.begin()];
    //     array<double, N_ITER + 1> pred_density = run_dft(best_grid);

    //     if (!write_grid(best_grid, grid_file)) { return 1; }
    //     if (!write_density(pred_density, density_file)) { return 1; }

    //     if (*min_cost_it+0.000000000000001 > last_cost) {
    //         cout << "No single cell improvements left" << endl;
    //         break;
    //     }
    //     last_cost = *min_cost_it;
    //     last_grid = grids[min_cost_it-costs.begin()];
    //     grids.fill(grids[min_cost_it-costs.begin()]);
    // }
    // cout << "================================================" << endl;
    // cout << "Final grid: " << endl;
    // for (int i = 0; i < N_SQUARES; ++i) {
    //     if (i % 20 == 0) { cout << endl << last_grid[i]; }
    //     else { cout << "," << last_grid[i]; }
    // }
    // cout << endl << endl;

    return 0;
}