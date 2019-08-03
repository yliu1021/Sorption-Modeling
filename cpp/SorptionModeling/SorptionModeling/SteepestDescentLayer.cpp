//
//  SteepestDescentLayer.cpp
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#include "SteepestDescentLayer.h"

#include <array>
#include <iostream>
#include <vector>

#include "Helpers.h"

using namespace std;

SteepestDescentLayer::SteepestDescentLayer() {
    SteepestDescentLayerOptions op;
    options_ = op;
}

SteepestDescentLayer::SteepestDescentLayer(SteepestDescentLayerOptions options) {
    options_ = options;
}

void SteepestDescentLayer::optimize() {
    if (!built()) {
        cerr << "ERROR: Tried to optimize layer without building it first" << endl;
        exit(1);
    }
    
    for (int g = 0; g < grids().size(); ++g) {
        
        cout << "Running Steepest Descent on grid " << g << " with a maximum of " << options_.max_iters << " iterations..." << endl;
        
        double min_cost = 1;
        array<Grid, N_SQUARES> toggled_grids;
        toggled_grids.fill(grids()[g]);
        Grid best_grid = grids()[g];
        Grid costs;

        for (int iterations = 0; iterations < options_.max_iters; iterations++) {
            for (int j = 0; j < N_SQUARES; ++j) {
                toggled_grids[j][j] = 1 - toggled_grids[j][j];
                costs[j] = mean_abs_error(target_curve(), run_dft(toggled_grids[j]));
            }
            double* min_cost_it = min_element(costs.begin(), costs.end());
            cout << "Minimum cost for iteration " << iterations << ": " << *min_cost_it << endl;

//            char grid_name[20];
//            sprintf(grid_name, "grid_%04d.csv", i);
//            char density_name[20];
//            sprintf(density_name, "density_%04d.csv", i);
//            string save_folder = "./evol_iter_grids/2/";
//            string grid_file = save_folder + grid_name;
//            string density_file = save_folder + density_name;
//            array<double, N_SQUARES> best_grid_iter = toggled_grids[min_cost_it - costs.begin()];
//            array<double, N_ITER+1> pred_density = run_dft(best_grid);
//            if (!write_grid(best_grid, grid_file)) { return 1; }
//            if (!write_density(pred_density, density_file)) { return 1; }

            if (*min_cost_it > min_cost) {
                cout << "No single cell improvements left" << endl;
                break;
            }
            min_cost = *min_cost_it;
            best_grid = toggled_grids[min_cost_it-costs.begin()];
            toggled_grids.fill(best_grid);
        }
        grids()[g] = best_grid;
    }
}
