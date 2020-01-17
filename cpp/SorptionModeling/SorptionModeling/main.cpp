//
//  main.cpp
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#include <iostream>
#include <array>

#include "OptimizationModel.h"

#include "SteepestDescentLayer.h"
#include "SwarmLayer.h"
#include "Helpers.h"

using namespace std;

constexpr int PARTITIONS = 6; // (horizontal lines)
constexpr double PARTITION_SIZE = 0.2;

int main(int argc, const char * argv[]) {
    
    vector<Grid> grids;
    grids.push_back(random_grid());
    vector<double> costs;

    OptimizationModel m;

    SwarmLayerOptions sl_1;
    sl_1.max_iters = 2500;
    SwarmLayer l1(sl_1);
    m.add_layer(l1);
    
    int curve_count = 0;
    
    array<double, N_ADSORP+1> target_curve;
    for (int p1 = 0; p1 < PARTITIONS; p1++) {
        for (int p2 = p1; p2 < PARTITIONS; p2++) {
            for (int p3 = p2; p3 < PARTITIONS; p3++) {
                for (int p4 = p3; p4 < PARTITIONS; p4++) {
                    for (int p5 = p4; p5 < PARTITIONS; p5++) {
                        for (int p6 = p5; p6 < PARTITIONS; p6++) {
                            
                            for (int i = 0; i < N_ADSORP+1; i++) {
                                if (i < (N_ADSORP+1)/5) {
                                    target_curve[i] = ((p2-p1)*PARTITION_SIZE)/(1.0/5) * (i*STEP_SIZE) + p1*PARTITION_SIZE;
                                } else if (i < 2*(N_ADSORP+1)/5) {
                                    target_curve[i] = ((p3-p2)*PARTITION_SIZE)/(1.0/5) * ((i*STEP_SIZE)-(1.0/5)) + p2*PARTITION_SIZE;
                                } else if (i < 3*(N_ADSORP+1)/5) {
                                    target_curve[i] = ((p4-p3)*PARTITION_SIZE)/(1.0/5) * ((i*STEP_SIZE)-(2.0/5)) + p3*PARTITION_SIZE;
                                } else if (i < 4*(N_ADSORP+1)/5) {
                                    target_curve[i] = ((p5-p4)*PARTITION_SIZE)/(1.0/5) * ((i*STEP_SIZE)-(3.0/5)) + p4*PARTITION_SIZE;
                                } else {
                                    target_curve[i] = ((p6-p5)*PARTITION_SIZE)/(1.0/5) * ((i*STEP_SIZE)-(4.0/5)) + p5*PARTITION_SIZE;
                                }
                                if (i == 0) { target_curve[i] = 0; }
                                if (i == N_ADSORP) { target_curve[i] = 1; }
                            }
                            cout << "CURVE COUNT: " << curve_count << endl;
                            cout << p1 << " " << p2 << " " << p3 << " " << p4 << " " << p5 << " " << p6 << endl;
                            
//                            if (p1 == 1 && p2 == 5 && p3 == 5 && p4 == 5 && p5 == 5 && p6 == 5) {
//                                for (int i = 0; i < N_ADSORP+1; i++) {
//                                    cout << i << ": " << target_curve[i] << endl;
//                                }
//                            }
                            
                            // DO STUFF HERE --------------------------------------------
                            
//                            FitOptions fo;
//                            fo.target_curve = target_curve;
//                            m.fit(fo, grids, costs);
//
//                            char grid_name[20];
//                            sprintf(grid_name, "grid_%04d.csv", curve_count);
//                            char density_name[20];
//                            sprintf(density_name, "density_%04d.csv", curve_count);
//                            string grid_save_folder = "./grids/";
//                            string results_save_folder = "./results/";
//                            string grid_file = grid_save_folder + grid_name;
//                            string density_file = results_save_folder + density_name;
//                            array<double, N_ITER+1> pred_density = run_dft(grids[0]);
//                            if (!write_grid(grids[0], grid_file)) { return 1; }
//                            if (!write_density(pred_density, density_file)) { return 1; }
                            
                            // END DO STUFF ------------------------------------------------
                            
                            grids.clear();
                            costs.clear();
                            grids.push_back(random_grid());
                            
                            curve_count++;
                        }
                    }
                }
            }
        }
    }
}
