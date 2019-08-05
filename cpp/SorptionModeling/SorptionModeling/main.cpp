//
//  main.cpp
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#include <iostream>

#include "OptimizationModel.h"

#include "SteepestDescentLayer.h"
#include "Helpers.h"

using namespace std;

int main(int argc, const char * argv[]) {
    OptimizationModel m;
    
    SteepestDescentLayerOptions sdl_1;
    sdl_1.max_iters = 5;
    SteepestDescentLayer l(sdl_1);
    m.add_layer(l);
    
    FitOptions fo;
    fo.target_curve = linear_curve();
    m.fit(5, fo); // Fit starting with 1 randomly initialized grids
}
