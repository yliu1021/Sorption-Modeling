//
//  OptimizationModel.cpp
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#include "OptimizationModel.h"

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
    std::vector<Grid> grids;
    for (int i = 0; i < n_grids; ++i) {
        grids.push_back(random_grid());
    }

    for (int i = 0; i < layers_.size(); ++i) {
        layers_[i]->init(&fit_options.target_curve, options_.cost_func, &grids);
        layers_[i]->optimize();
    }
}
