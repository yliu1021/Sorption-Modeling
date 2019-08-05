//
//  OptimizationModel.h
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#ifndef OptimizationModel_h
#define OptimizationModel_h

#include <array>
#include <string>
#include <vector>

#include "Constants.h"
#include "Helpers.h"

class OptimizationLayer;

struct ModelOptions {
    CostFunction cost_func = &mean_abs_error;
    bool population = false; // Whether to treat grids individually or together as a population
    bool repeat = false;
    bool repeat_iters = 3;
};

struct FitOptions {
    std::array<double,N_ADSORP+1> target_curve;
    bool verbose = false;
};

class OptimizationModel {
public:
    OptimizationModel();
    OptimizationModel(ModelOptions options);
    
    void add_layer(OptimizationLayer &layer);

    void fit(FitOptions fit_options, std::vector<Grid> &grids, std::vector<double> &costs);
    

private:    
    ModelOptions options_;
    std::vector<OptimizationLayer*> layers_;
};

#endif /* OptimizationModel_h */
