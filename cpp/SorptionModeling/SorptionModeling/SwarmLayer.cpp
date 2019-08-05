//
//  SwarmLayer.cpp
//  SorptionModeling
//
//  Created by Kevin Li on 8/5/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#include "SwarmLayer.h"

#include <iostream>
#include <math.h>


#include "Constants.h"
#include "Helpers.h"

using namespace std;

SwarmLayer::SwarmLayer() {
    SwarmLayerOptions op;
    options_ = op;
}

SwarmLayer::SwarmLayer(SwarmLayerOptions options) {
    options_ = options;
}

void SwarmLayer::optimize() {
    if (!built()) {
        cerr << "ERROR: Tried to optimize layer without building it first" << endl;
        exit(1);
    }
    
    cout << "Running swarm on grid with a maximum of " << options_.max_iters << " iterations..." << endl;

    while (grids().size() < options_.bees) {
        Grid g = random_grid();
        grids().push_back(g);
        costs().push_back((*cost_func())(target_curve(), run_dft(g)));
    }
    
    vector<Grid> velocities;
    vector<Grid> personal_best_grids(grids());
    Grid global_best_grid = grids()[min_element(costs().begin(), costs().end()) - costs().begin()];
    vector<double> personal_min_cost(costs());
    double global_min_cost = costs()[min_element(costs().begin(), costs().end()) - costs().begin()];;
    
    // Initialize random velocities
    for (int i = 0; i < grids().size(); ++i) {
        Grid v_grid;
        for (int j = 0; j < N_SQUARES; ++j) {
            double v = ((double)rand()/(RAND_MAX)) * 0.995 + 0.0025;
            v_grid[j] = sigmoid_inverse(v);
        }
        velocities.push_back(v_grid);
    }
    
    for (int iteration = 0; iteration < options_.max_iters; ++iteration) {
        for (int i = 0; i < grids().size(); ++i) {
            if (costs()[i] < personal_min_cost[i]) {
                personal_min_cost[i] = costs()[i];
                personal_best_grids[i] = grids()[i];
            }
            if (costs()[i] < global_min_cost) {
                global_min_cost = costs()[i];
                global_best_grid = grids()[i];
            }
        }
        
//        double vsum = 0;
        // Update velocities
        double r1 = ((double)rand()/(RAND_MAX));
        double r2 = ((double)rand()/(RAND_MAX));
        for (int i = 0; i < grids().size(); ++i) {
            for (int j = 0; j < N_SQUARES; ++j) {
                double p_contrib = 0.5 * r1 * (personal_best_grids[i][j] - grids()[i][j]);
                double g_contrib = 0.5 * r2 * (global_best_grid[j] - grids()[i][j]);
                velocities[i][j] += (p_contrib + g_contrib);
                
                // cout << p_contrib << "\t\t\t" << g_contrib << endl;
                
                // Clip velocities
                velocities[i][j] = max(velocities[i][j], options_.v_min);
                velocities[i][j] = min(velocities[i][j], options_.v_max);
//                vsum += abs(velocities[i][j]);
            }
        }
        if (iteration % 20 == 0) {
//            cout << "AVERAGE VELOCITY: " << vsum/(grids().size()*N_SQUARES) << endl;
        }
        
        // Update positions
        for (int i = 0; i < grids().size(); ++i) {
            for (int j = 0; j < N_SQUARES; ++j) {
                if (((double)rand()/(RAND_MAX)) < sigmoid(velocities[i][j])) {
                    // if (positions[i][j] != 0) { cout << "!"; }
                    grids()[i][j] = 1;
                } else {
                    // if (positions[i][j] != 1) { cout << "!"; }
                    grids()[i][j] = 0;
                }
            }
        }
        
        compute_costs();
        if (iteration % 20 == 0) {
            cout << "Minimum cost at iteration " << iteration << ": " << *min_element(costs().begin(), costs().end()) << endl;
        }
    }
}
