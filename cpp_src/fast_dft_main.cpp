#include "fast_dft.cpp"

/*
fast_dft usage:
	fast_dft ...folder/containing/grid/folder/
*/

int main(int argc, char *argv[]) {
	setup_NL();
	if (argc == 1) {
		double *grid = load_grid(cin);
		double *density = run_dft(grid);
		
		write_density(density, cout);
		
		delete[] grid;
		delete[] density;
	} else if (argc == 2) {
		string base_dir(argv[1]);
		if (base_dir.back() != '/')
			base_dir = base_dir + "/";
		// base_dir = "./generative_model/step##/"
		string grid_dir = base_dir + "grids/";
		string density_dir = base_dir + "results/";
		string cmd = "mkdir -p " + density_dir;
		system(cmd.c_str());

		for (int i = 0; true; ++i) {
			char grid_name[20];
			sprintf(grid_name, "grid_%04d.csv", i);
			char density_name[20];
			sprintf(density_name, "density_%04d.csv", i);
		
			string grid_file = grid_dir + grid_name;
			string density_file = density_dir + density_name;
		
			double *grid = load_grid(grid_file);
			if (grid == nullptr) break;
			double *pred_density = run_dft(grid);
			if (pred_density == nullptr) return -1;
			
			bool write_success = write_density(pred_density, density_file);
			if (!write_success) return -1;

			delete[] grid;
			delete[] pred_density;
		}
	} else {
		cerr << "Invalid cmd line arguments" << endl;
	}
	return 0;
}
