//
//  OptimizationLayer.h
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#ifndef OptimizationLayer_h
#define OptimizationLayer_h

#include <array>
#include <vector>
#include <thread>

#include "Constants.h"

/*
 * Abstract base layer class
 */
class OptimizationLayer {
public:
    // Construction and Initialization
    OptimizationLayer(bool verbose = false);
    virtual void init(std::array<double,N_ADSORP+1> *target_curve,
                      double (*cost_func)(const std::array<double,N_ADSORP+1>&, const std::array<double,N_ITER+1>&),
                      std::vector<Grid> *grids,
                      std::vector<double> *costs);
    
    // Getters
    bool verbose() const { return verbose_; }
    bool built() const { return built_; };
    const std::array<double,N_ADSORP+1>& target_curve() const { return *target_curve_; }
    CostFunction cost_func() const { return cost_func_; }
    
    std::vector<Grid>& grids() { return *grids_; }
    std::vector<double>& costs() { return *costs_; }
    
    // Functions
    virtual void compute_costs();
    virtual void optimize() = 0;

private:
    void compute_costs_thread(int threadID);
    
    bool verbose_;
    bool built_ = false;
    
    std::array<double,N_ADSORP+1> *target_curve_;
    CostFunction cost_func_;
    std::vector<Grid> *grids_;
    std::vector<double> *costs_;
};

#endif /* OptimizationLayer_h */
