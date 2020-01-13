#include <array>
#include <iostream>
// #include <fstream>
// #include <vector>
#include <string>
#include <math.h>
#include <thread>

// #include <ctime>
// #include <cstring>
// #include <cmath>
// #include <cstdio>
// #include <cstdlib>

using namespace std;

#include <sys/stat.h>

#include "constants.h"
#include "helpers.h"

constexpr int NUM_THREADS = 3;
constexpr int BEES = 25;
constexpr int ITERS = 100;
string save_path = "./swarm_grids/";

string dir;

double sigmoid(double x) {
    return (1 / (1 + exp(-x)));
}

double sigmoid_inverse(double x) {
    return log(x / (1-x));
}

bool file_exists (const string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

void thread_func() {
    // Hyperparameters
    double v_min = sigmoid_inverse(0.0025);
    double v_max = sigmoid_inverse(0.9975);

    if (dir[dir.size()-1] != '/') {
        dir.append("/");
    }

    while (true) {
        int current, attempts = 0;
        string grid_file, density_file;
        do {
            current = rand() % 10000;
            char grid_name[20];
            sprintf(grid_name, "grid_%04d.csv", current);
            char density_name[20];
            sprintf(density_name, "density_%04d.csv", current);
            grid_file = dir + "grids/" + grid_name;
            density_file = dir + "results/" + density_name;
            if (attempts++ > 50000) { return; }
        } while (!(file_exists(density_file) && !file_exists(grid_file)));

        // string path = "../data_generation/results/density_0000.csv";
        array<double,N_ADSORP+1> target_curve = load_density(density_file);
        // for (int i = 0; i < N_ADSORP+1; ++i) {
        //     cout << target_curve[i] << endl;
        // }

        array<Grid, BEES> positions;
        array<Grid, BEES> velocities;
        array<double, BEES> costs;

        // Initialize random positions and velocities
        for (int i = 0; i < BEES; ++i) {
            positions[i] = random_grid();
            costs[i] = mean_abs_error(target_curve, run_dft(positions[i]));
            for (int j = 0; j < N_SQUARES; ++j) {
                double v = ((double)rand()/(RAND_MAX)) * 0.995 + 0.0025;
                velocities[i][j] = sigmoid_inverse(v);
            }
        }
        
        array<Grid, BEES> personal_best_grids(positions);
        Grid global_best_grid = positions[min_element(costs.begin(), costs.end()) - costs.begin()];
        array<double, BEES> personal_min_cost(costs);
        double global_min_cost = costs[min_element(costs.begin(), costs.end()) - costs.begin()];

        for (int iteration = 0; iteration < ITERS; ++iteration) {
            if (iteration % 10 == 0) {
                cout << "Minimum cost at start of iteration " << iteration << ": " << global_min_cost << endl;
            }

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
                }
            }

            // Update positions
            for (int i = 0; i < BEES; ++i) {
                for (int j = 0; j < N_SQUARES; ++j) {
                    if (((double)rand()/(RAND_MAX)) < sigmoid(velocities[i][j])) {
                        positions[i][j] = 1;
                    } else {
                        positions[i][j] = 0;
                    }
                }
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
        }

        // Save best grid
        if (!write_grid(global_best_grid, grid_file)) { return; }
    }
}

int main(int argc, char *argv[]) {
    if (argc == 2 && file_exists(argv[1])) {
        dir = argv[1];
    } else {
        cerr << "USAGE: ./swarm DIRECTORY" << endl;
        return 1;
    }

	setup_NL(); // for fast_dft
    srand(time(NULL)); // Generate random seed for random number generation

    std::array<std::thread, NUM_THREADS> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads[i] = thread(thread_func);
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads[i].join();
    }

    return 0;

}