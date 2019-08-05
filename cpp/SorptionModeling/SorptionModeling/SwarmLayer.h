//
//  SwarmLayer.h
//  SorptionModeling
//
//  Created by Kevin Li on 8/5/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#ifndef SwarmLayer_h
#define SwarmLayer_h

#include "Helpers.h"
#include "OptimizationLayer.h"

struct SwarmLayerOptions {
    int max_iters = INT_MAX;
    int bees = 20;
    double v_max = sigmoid_inverse(0.9975);
    double v_min = sigmoid_inverse(0.0025);
    bool verbose = true;
};

class SwarmLayer : public OptimizationLayer {
public:
    SwarmLayer();
    SwarmLayer(SwarmLayerOptions options);
    
    virtual void optimize();
    
private:
    SwarmLayerOptions options_;
};

#endif /* SwarmLayer_h */
