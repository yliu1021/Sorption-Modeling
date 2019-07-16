//
//  DFT.swift
//  GuiLearning
//
//  Created by Yuhan Liu on 7/16/19.
//  Copyright © 2019 Yuhan Liu. All rights reserved.
//

import Foundation

let GRID_SIZE = 20
let N_SQUARES = GRID_SIZE * GRID_SIZE
let N_ITER = 80
let N_ADSORP = 40
let STEP_SIZE = 0.025

let KB = 0.0019872041
let T = 298.0
let Y = 1.5
let TC = 647.0
let BETA = 1 / (KB * T);
let MUSAT = -2.0 * KB * TC;
let C = 4.0;
let WFF = -2.0 * MUSAT / C;

let row = [Int](repeating: 0, count: N_SQUARES)
var NL: [[Int]] = [[Int]](repeating: row, count: N_SQUARES + 1)

func runDFT(grid: [Double]) -> [Double] {
    var r0 = [Double](repeating: 0, count: N_SQUARES + 1)
    var r1 = r0
    
    var Ntotal_pores = 0.0
    for i in 1...N_SQUARES {
        let g = grid[i - 1]
        Ntotal_pores += g
        r0[i] = g
    }
    guard Ntotal_pores > 0.01 else {
        return [Double](repeating: 0, count: N_ITER + 1)
    }
    
    for i in 1...N_SQUARES {
        r1[i] = grid[i - 1]
    }
    
    var density = [Double](repeating: 0, count: N_ITER + 1)
    
    for jj in 0...N_ITER {
        let RH: Double
        let muu: Double
        if jj <= N_ADSORP {
            RH = Double(jj) * STEP_SIZE
        } else {
            RH = Double(N_ADSORP) * STEP_SIZE - (jj - N_ADSORP) * STEP_SIZE
        }
        if RH != 0.0 {
            muu = MUSAT + KB*T*log(RH)
        } else {
            muu = -90.0
        }
        
        for _ in 1..<100000000 {
            var vi = [Double](repeating: 0, count: N_SQUARES + 1)
            for i in 1...N_SQUARES {
                let a1 = NL[i][1]
                let a2 = NL[i][2]
                let a3 = NL[i][3]
                let a4 = NL[i][4]
                vi[i] = WFF * (r1[a1] + Y*(1 - r0[a1])) +
                        WFF * (r1[a2] + Y*(1 - r0[a2])) +
                        WFF * (r1[a3] + Y*(1 - r0[a3])) +
                        WFF * (r1[a4] + Y*(1 - r0[a4])) +
                        muu
            }
            var power_drou = 0.0
            var rounew = [Double](repeating: 0, count: N_SQUARES + 1)
            for i in 0...N_SQUARES {
                rounew[i] = r0[i] / (1 + exp(-BETA * vi[i]))
            }
            for i in 0..<N_SQUARES {
                let diff = rounew[i] - r1[i]
                power_drou += diff * diff;
                r1[i] = rounew[i]
            }
            if power_drou < 1e-10 * N_SQUARES {
                break
            }
        }
        density[jj] = r1.reduce(0, +) / Ntotal_pores
    }
    return density
}

func runDFT(grid: [[Double]]) -> [Double] {
    let flattendGrid = grid.flatMap{ $0 }
    return runDFT(grid: flattendGrid)
}

func setupNL() {
    var r0 = [Int](repeating: 0, count: N_SQUARES + 1)
    var r1 = r0
    
    var L = 0
    for i in 1...N_SQUARES {
        r0[i] = L
        if i % GRID_SIZE == 0 {
            L = 0
        } else {
            L += 1
        }
    }
    L = 0
    for i in 1...N_SQUARES {
        r1[i] = L
        if i % GRID_SIZE == 0 {
            L += 1
        }
    }
    
    var NN = [Int](repeating: 0, count: N_SQUARES + 1)
    for i in 1..<N_SQUARES {
        for jj in (i+1)...N_SQUARES {
            var r11 = Double(r0[jj] - r0[i])
            var r12 = Double(r1[jj] - r1[i])
            r11 = r11 - round(r11 / Double(GRID_SIZE)) * GRID_SIZE
            r12 = r12 - round(r12 / Double(GRID_SIZE)) * GRID_SIZE
            let d12_square = r11 * r11 + r12 * r12
            if (d12_square <= 1.01 * 1.01) {
                NN[i] += 1
                NN[jj] += 1
                NL[i][NN[i]] = jj
                NL[jj][NN[jj]] = i
            }
        }
    }
}
