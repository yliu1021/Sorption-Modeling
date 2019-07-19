#include <array>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <thread>

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;

#include "constants.h"
#include "fast_dft_std.cpp"
#include "helpers.cpp"

int main(int argc, char *argv[]) {
	setup_NL(); // for fast_dft

    int ITERS = 5;

    if (argc == 2) {
        string grid_path(argv[1]);
        std::array<double, N_SQUARES> grid = load_grid(grid_path);

        for (int i = 0; i < ITERS; ++i) {
            std::array<double, N_ITER+1> pred_density = run_dft(grid);

            std::array<double, N_ADSORP+1> linear = linear_curve();
        //     // kullback_leibler_divergence(linear, pred_density);
        //     // cout << "COST: " << kullback_leibler_divergence(linear, pred_density) << endl;
            cout << "COST: " << mean_abs_error(linear, pred_density) << endl;
            toggle_random(grid);
        }
    } else {
        cerr << "Invalid cmd line arguments" << endl;
    }

    return 0;
}