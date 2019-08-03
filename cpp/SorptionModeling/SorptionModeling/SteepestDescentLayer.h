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
    bool verbose = false;
};

class SteepestDescentLayer : public OptimizationLayer {
public:
    SteepestDescentLayer();
    SteepestDescentLayer(SteepestDescentLayerOptions options);
    
    virtual void optimize();
    
private:
    SteepestDescentLayerOptions options_;
};

#endif /* SteepestDescentLayer_h */
