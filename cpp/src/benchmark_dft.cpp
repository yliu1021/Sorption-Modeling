#include <iostream>
#include <string>
#include <array>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

#include "helpers.h"
#include "constants.h"

using namespace std;

array<double, N_SQUARES> rand_grid(uniform_real_distribution<double> &unif, default_random_engine &re) {
	array<double, N_SQUARES> grid;
	for (int i = 0; i < N_SQUARES; ++i) {
		double rand_num = unif(re);
		if (rand_num < 0.5) {
			grid[i] = 1.0;
		} else {
			grid[i] = 0.0;
		}
	}
	return grid;
}

vector<array<double, N_SQUARES>> gen_grids(int num_grids) {
	vector<array<double, N_SQUARES>> grids;

	unsigned seed1 = chrono::system_clock::now().time_since_epoch().count();
    uniform_real_distribution<double> unif(0, 1);
    default_random_engine re(seed1);

	for (int i = 0; i < num_grids; ++i) {
		grids.push_back(rand_grid(unif, re));
	}

	return grids;
}

double max_abs_diff(array<double, N_ITER + 1> &arr1, array<double, N_ITER + 1> &arr2) {
	double max_diff = 0.0;
	for (int i = 0; i <= N_ITER; ++i) {
		double diff = abs(arr1[i] - arr2[i]);
		if (max_diff < diff) max_diff = diff;
	}
	return max_diff;
}

int main() {
	setup_NL();

	// constexpr int num_grids = 400;
	//
	// vector<array<double, N_SQUARES>> grids = gen_grids(num_grids);
	// vector<array<double, N_ITER + 1>> result1;
	// result1.reserve(num_grids);
	// vector<array<double, N_ITER + 1>> result2;
	// result2.reserve(num_grids);
	//
	// auto start1 = chrono::high_resolution_clock::now();
	// for (int i = 0; i < num_grids; ++i) {
	// 	result1.push_back(run_dft(grids[i]));
	// }
	// auto end1 = chrono::high_resolution_clock::now();
	// auto d1 = chrono::duration_cast<chrono::microseconds>(end1 - start1);
	// double duration1 = d1.count();
	// cout << "Normal version: " << setprecision(6) << duration1/1000000 << endl;
	//
	// auto start2 = chrono::high_resolution_clock::now();
	// for (int i = 0; i < num_grids; ++i) {
	// 	result2.push_back(run_dft_fast(grids[i]));
	// }
	// auto end2 = chrono::high_resolution_clock::now();
	// auto d2 = chrono::duration_cast<chrono::microseconds>(end2 - start2);
	// double duration2 = d2.count();
	// cout << "Optimized version: " << setprecision(6) << duration2/1000000 << endl;
	//
	// cout << "Speed up: " << setprecision(3) << (1 - duration2/duration1)*100 << "%" << endl;
	//
	// double max_diff = 0.0;
	// for (int i = 0; i < num_grids; ++i) {
	// 	double diff = max_abs_diff(result1[i], result2[i]);
	// 	if (max_diff < diff) max_diff = diff;
	// }
	//
	// cout << "max abs diff: " << max_diff << endl;
	
	return 0;
}