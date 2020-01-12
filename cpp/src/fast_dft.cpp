#include <sys/stat.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>
#include <string>
#include <thread>

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "constants.h"
#include "helpers.h"

using namespace std;
/*
  fast_dft usage:
  fast_dft ...folder/containing/grid/folder/
*/

void process_files(const string &grid_dir, const string &density_dir,
                   int from, int to) { // [from, to)
  for (int i = from; i < to; ++i) {
    char grid_name_c[40];
    sprintf(grid_name_c, "grid_%04d.csv", i);
    char density_name_c[40];
    sprintf(density_name_c, "density_%04d.csv", i);
    string grid_name(grid_name_c);
    string density_name(density_name_c);
    string grid_file = grid_dir + grid_name;
    string density_file = density_dir + density_name;
    array<double,N_SQUARES> grid = load_grid(grid_file);
    array<double,N_ADSORP+1> pred_density = run_dft(grid);
    
    bool write_success = write_density(pred_density, density_file);
    if (!write_success) return;
  }
}

int main(int argc, char *argv[]) {
  setup_NL();
  if (argc == 1) {
    array<double,N_SQUARES> grid = load_grid(cin);
    array<double,N_ADSORP+1> density = run_dft(grid);
    
    write_density(density, cout);
    
  } else if (argc == 2) {
    string base_dir(argv[1]);
    if (base_dir.back() != '/')
      base_dir = base_dir + "/";
    // base_dir = "./generative_model/step##/"
    string grid_dir = base_dir + "grids/";
    string density_dir = base_dir + "results/";
    string cmd = "mkdir -p " + density_dir;
    system(cmd.c_str());
    // count number of grid files
    int num_grid_files;
    for (int i = 0; true; ++i) {
      char grid_name_c[40];
      sprintf(grid_name_c, "grid_%04d.csv", i);
      string grid_name(grid_name_c);
      string grid_file = grid_dir + grid_name;
      struct stat buf;
      if (stat(grid_file.c_str(), &buf) != 0) {
        num_grid_files = i;
        break;
      }
    }
    vector<thread*> threads;
    const int NUM_THREADS = 16;
    int start = 0;
    int step = num_grid_files / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; ++i) {
      thread *t = new thread(process_files, grid_dir, density_dir, start, start+step);
      start += step;
      threads.push_back(t);
    }
    if (start < num_grid_files) {
      thread *t = new thread(process_files, grid_dir, density_dir, start, num_grid_files);
      threads.push_back(t);
    }
    for (auto t : threads) {
      t->join();
      delete t;
    }
  } else {
    cerr << "Invalid cmd line arguments" << endl;
  }
  return 0;
}
