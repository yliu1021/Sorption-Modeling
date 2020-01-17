//
//  SteepestDescentLayer.h
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#ifndef SteepestDescentLayer_h
#define SteepestDescentLayer_h

#include <array>
#include <vector>

#include "OptimizationLayer.h"
#include "Constants.h"

struct SteepestDescentLayerOptions {
    int max_iters = INT_MAX;
    bool verbose = true;
};

class SteepestDescentLayer : public OptimizationLayer {
public:
    SteepestDescentLayer();
    SteepestDescentLayer(SteepestDescentLayerOptions options);
    
    virtual void optimize();
    
private:
    void compute_costs_thread(int thread_num, std::array<Grid, N_SQUARES> &toggled_grids, std::array<double, N_SQUARES> &costs);
    
    SteepestDescentLayerOptions options_;
};

#endif /* SteepestDescentLayer_h */
