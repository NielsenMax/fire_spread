#include <iostream>
#include <string>

#include "ignition_cells.hpp"
#include "landscape.hpp"
#include "many_simulations.hpp"
#include "spread_functions.hpp"

#define DISTANCE 30.0f
#define ELEVATION_MEAN 1163.3f
#define ELEVATION_SD 399.5f
#define UPPER_LIMIT 0.5f
#define SIMULATIONS 100.0f

int main(int argc, char* argv[]) {
  try {

    // check if the number of arguments is correct
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <landscape_file_prefix>" << std::endl;
      return EXIT_FAILURE;
    }

    // read the landscape file prefix
    std::string landscape_file_prefix = argv[1];

    // read the landscape
    Landscape landscape(
        landscape_file_prefix + "-metadata.csv", landscape_file_prefix + "-landscape.csv"
    );

    // read the ignition cells
    IgnitionCells ignition_cells =
        read_ignition_cells(landscape_file_prefix + "-ignition_points.csv");

    SimulationParams params = {
      0.0f, 0.5f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f
    };

    Matrix<size_t> burned_amounts = burned_amounts_per_cell(
        landscape, ignition_cells, params, DISTANCE, ELEVATION_MEAN, ELEVATION_SD, UPPER_LIMIT,
        SIMULATIONS
    );
    std::cout << "Landscape size: " << landscape.width << " " << landscape.height << std::endl;
    std::cout << "Simulations: " << SIMULATIONS << std::endl;
    for (size_t i = 0; i < landscape.height; i++) {
      for (size_t j = 0; j < landscape.width; j++) {
        if (j != 0) {
          std::cout << " ";
        }
        std::cout << burned_amounts[{j, i}];
      }
      std::cout << std::endl;
    }

  } catch (std::runtime_error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}