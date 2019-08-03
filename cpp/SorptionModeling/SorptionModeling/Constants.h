//
//  Constants.h
//  SorptionModeling
//
//  Created by Kevin Li on 8/2/19.
//  Copyright © 2019 PARISLab. All rights reserved.
//

#ifndef Constants_h
#define Constants_h

constexpr int GRID_SIZE = 20;
constexpr int N_SQUARES = (GRID_SIZE * GRID_SIZE);
constexpr int N_ITER = 80;
constexpr int N_ADSORP = 40;
constexpr double STEP_SIZE = 0.025;

typedef std::array<double, N_SQUARES> Grid;
typedef double (*CostFunction)(const std::array<double,N_ADSORP+1>&, const std::array<double,N_ITER+1>&);

#endif /* Constants_h */
