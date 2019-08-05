//
//  OptimizationModel.cpp
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#include "OptimizationModel.h"

//#include <thread>

#include "Helpers.h"
#include "OptimizationLayer.h"

using namespace std;

OptimizationModel::OptimizationModel() {
    ModelOptions op;
    options_ = op;
    setup_NL();
}

OptimizationModel::OptimizationModel(ModelOptions options) {
    options_ = options;
    setup_NL();
}

void OptimizationModel::add_layer(OptimizationLayer &layer) {
    layers_.push_back(&layer);
}

void OptimizationModel::fit(int n_grids, FitOptions fit_options) {
    cout << "Starting optimization using " << NUM_THREADS << " threads. " << endl;
//    cout << "This machine supports " << thread::hardware_concurrency() << " threads." << endl;
    
    std::vector<Grid> grids;
    std::vector<double> costs;
    for (int i = 0; i < n_grids; ++i) {
        Grid g = random_grid();
        grids.push_back(g);
        costs.push_back((options_.cost_func)(fit_options.target_curve, run_dft(g)));
    }

    for (int i = 0; i < layers_.size(); ++i) {
        layers_[i]->init(&fit_options.target_curve, options_.cost_func, &grids, &costs);
        layers_[i]->optimize();
    }
}
