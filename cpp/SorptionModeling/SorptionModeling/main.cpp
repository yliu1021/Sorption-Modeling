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
    for (int i = 0; i < 3; ++i) {
        grids.push_back(random_grid());
    }
    vector<double> costs;
    
    OptimizationModel m;
    
//    SwarmLayerOptions sl_1;
//    sl_1.max_iters = 101;
//    SwarmLayer l1(sl_1);
//    m.add_layer(l1);

    SteepestDescentLayerOptions sdl_1;
    sdl_1.max_iters = 1;
    SteepestDescentLayer l2(sdl_1);
    m.add_layer(l2);
    
    FitOptions fo;
    fo.target_curve = linear_curve();
    m.fit(fo, grids, costs);

    write_grid(grids[min_element(costs.begin(), costs.end()) - costs.begin()], cout);
    cout << "\nmin cost: " << *min_element(costs.begin(), costs.end()) << endl;
    
    for (int i = 0; i < grids.size(); i++) {
        char grid_name[20];
        sprintf(grid_name, "grid_%04d.csv", i);
        char density_name[20];
        sprintf(density_name, "density_%04d.csv", i);
        string save_folder = "./optimal_grids/";
        string grid_file = save_folder + grid_name;
        string density_file = save_folder + density_name;
        array<double, N_ITER+1> pred_density = run_dft(grids[i]);
        if (!write_grid(grids[i], grid_file)) { return 1; }
        if (!write_density(pred_density, density_file)) { return 1; }
    }
}
