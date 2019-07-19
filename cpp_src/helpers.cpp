#include <array>
#include <math.h>
#include <stdlib.h>
#include<iterator>

#include "constants.h"

#define EPSILON 0.0000001

void toggle_random(std::array<double, N_SQUARES> &grid) {
    int sq = rand() % N_SQUARES;
    grid[sq] = (grid[sq] == 0) ? 1 : 0;
}

// Normalize the values in an array to between 0 and 1
void normalizeArr(std::vector<double>::iterator begin, std::vector<double>::iterator end, double min, double max) {
    // TODO: Make normalized STDDEV
     double diff = max - min;
     if (diff == 0) { diff = 1; } // If all grids have same cost

     // print each value held in the array
     while (begin != end) {
          *(begin) -= min;
          *(begin) /= diff;
          begin++;
     }
}

// Normalize the values in an array to between 0 and 1
void normalizeArr(double* piStart, double* piLast, double min, double max){

     // Calculate the size of the array (how many values it holds)
     unsigned int uiArraySize = piLast - piStart;

     double diff = max - min;
     if (diff == 0) { diff = 1; } // If all grids have same cost

     // print each value held in the array
     for (unsigned int uiCount = 0; uiCount < uiArraySize; uiCount++) {
          *(piStart + uiCount) -= min;
          *(piStart + uiCount) /= diff;
     }
}


// /*
// Clip (limit) the values in an array.
// Given an interval, values outside the interval are clipped to the interval 
// edges. For example, if an interval of [0, 1] is specified, values smaller than 
// 0 become 0, and values larger than 1 become 1. 
// */
// void clip(double a[], double a_min, double a_max, unsigned short len=N_ADSORP+1) {
//     for (unsigned short i = 0; i < len; ++i) {
//         a[i] = std::max(a[i], a_min);
//         a[i] = std::min(a[i], a_min);
//     }
// }

void clip(std::array<double, N_ADSORP+1> a, const double a_min, const double a_max) {
    for (unsigned short i = 0; i < a.size(); ++i) { 
        a[i] = std::max(a[i], a_min);
        a[i] = std::min(a[i], a_max);
    } 
}

// double kullback_leibler_divergence(const std::array<double, N_ADSORP+1> y_true, const std::array<double, N_ITER+1> y_pred) {
//     // TODO: Something wrong-- NAN values
//     clip(y_true, EPSILON, 1);
//     clip(y_pred, EPSILON, 1);
//     double sum = 0;
//     for (unsigned short i = 0; i < N_ADSORP+1; ++i) {
//         cout << y_true[i] << "\t";
//         cout << y_pred[i] << "\t";
//         cout << y_true[i]/y_pred[i] << endl;
//         sum += y_true[i] * log(y_true[i] / y_pred[i]);
//     }
//     cout << "SUM: " << sum << endl;
//     return sum;
// }

double mean_abs_error(const std::array<double, N_ADSORP+1> y_true, const std::array<double, N_ITER+1> y_pred) {
    double mse = 0;
    for (unsigned short i = 0; i < N_ADSORP+1; ++i) {
        mse += abs(y_true[i] - y_pred[i]);
    }
    mse /= (N_ADSORP+1);
    return mse;
}

// double kullback_leibler_divergence(double y_true[], double y_pred[], unsigned short len=N_ADSORP+1) {
//     clip(y_true, EPSILON, 1);
//     clip(y_pred, EPSILON, 1);
//     double sum = 0;
//     for (unsigned short i = 0; i < len; ++i) {
//         sum += y_true[i] * log(y_true[i] / y_pred[i]);
//     }
//     return sum;
// }

std::array<double, N_ADSORP+1> linear_curve() {
    std::array<double, N_ADSORP+1> lin;
    double v = 0;
    for (unsigned short i = 0; i < N_ADSORP+1; ++i, v += STEP_SIZE) {
        lin[i] = v;
    }
    return lin;
}