//
//  OptimizationLayer.cpp
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#include "OptimizationLayer.h"

OptimizationLayer::OptimizationLayer(bool verbose) {
    verbose_ = verbose;
}

void OptimizationLayer::init(std::array<double,N_ADSORP+1> *target_curve,
                             double (*cost_func)(const std::array<double,N_ADSORP+1>&, const std::array<double,N_ITER+1>&),
                             std::vector<Grid> *grids) {
    target_curve_ = target_curve;
    cost_func_ = cost_func;
    grids_ = grids;
    built_ = true;
}
