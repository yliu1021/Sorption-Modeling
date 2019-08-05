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
#include "SwarmLayer.h"
#include "Helpers.h"

using namespace std;

int main(int argc, const char * argv[]) {
    
    vector<Grid> grids;
    for (int i = 0; i < 5; ++i) {
        grids.push_back(random_grid());
    }
    vector<double> costs;
    
    OptimizationModel m;
    
//    SteepestDescentLayerOptions sdl_1;
//    sdl_1.max_iters = 5;
//    SteepestDescentLayer l(sdl_1);
//    m.add_layer(l);
    
    SwarmLayerOptions sl_1;
    sl_1.max_iters = 1001;
    SwarmLayer l(sl_1);
    m.add_layer(l);
    
    FitOptions fo;
    fo.target_curve = linear_curve();
    m.fit(fo, grids, costs); // Fit starting with 5 randomly initialized grids
    
//    for (int i = 0; i < 3; ++i) {
//        cout << costs[i] << endl;
//    }
    write_grid(grids[min_element(costs.begin(), costs.end()) - costs.begin()], cout);
    cout << "\nmin cost: " << *min_element(costs.begin(), costs.end()) << endl;
}
