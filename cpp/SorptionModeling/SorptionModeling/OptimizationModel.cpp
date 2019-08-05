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

void OptimizationModel::fit(FitOptions fit_options, vector<Grid> &grids, vector<double> &costs) {
    cout << "Starting optimization using " << NUM_THREADS << " threads. " << endl;
    
    srand(static_cast<unsigned int>(time(NULL)));
    
    costs.clear();
    for (int i = 0; i < grids.size(); ++i) {
        costs.push_back((options_.cost_func)(fit_options.target_curve, run_dft(grids[i])));
    }
    
    if (options_.population) {
        for (int i = 0; i < layers_.size(); ++i) {
            layers_[i]->init(&fit_options.target_curve, options_.cost_func, &grids, &costs);
            layers_[i]->optimize();
        }
    } else {
        for (int i = 0; i < grids.size(); i++) {
            cout << "Optimizing grid " <<  i << "..." << endl;
            
            vector<Grid> g;
            vector<double> g_costs;
            g.push_back(grids[i]);
            g_costs.push_back(costs[i]);
            for (int i = 0; i < layers_.size(); ++i) {
                layers_[i]->init(&fit_options.target_curve, options_.cost_func, &g, &g_costs);
                layers_[i]->optimize();
            }
            grids[i] = g[min_element(g_costs.begin(), g_costs.end()) - g_costs.begin()];
            costs[i] = *min_element(g_costs.begin(), g_costs.end());
        }
    }
}
