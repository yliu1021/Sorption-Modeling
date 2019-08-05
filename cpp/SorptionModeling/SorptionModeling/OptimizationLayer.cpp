//
//  OptimizationLayer.cpp
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#include "OptimizationLayer.h"

#include "Helpers.h"

OptimizationLayer::OptimizationLayer(bool verbose) {
    verbose_ = verbose;
}

void OptimizationLayer::init(std::array<double,N_ADSORP+1> *target_curve,
                             double (*cost_func)(const std::array<double,N_ADSORP+1>&, const std::array<double,N_ITER+1>&),
                             std::vector<Grid> *grids,
                             std::vector<double> *costs) {
    target_curve_ = target_curve;
    cost_func_ = cost_func;
    grids_ = grids;
    costs_ = costs;
    built_ = true;
}

void OptimizationLayer::compute_costs() {
    std::array<std::thread, NUM_THREADS> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads[i] = std::thread(&OptimizationLayer::compute_costs_thread, this, i);
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads[i].join();
    }
}

void OptimizationLayer::compute_costs_thread(int threadID) {
    for (int i = threadID; i < (*grids_).size(); i += NUM_THREADS) {
        costs()[i] = (*cost_func_)(target_curve(), run_dft(grids()[i]));
    }
}
