// filepath: graphics/fire_animation_data.cpp

#include <ctime> // Needed for time(0) seed
#include <iostream>
#include <random>

#include "fires.hpp"
#include "ignition_cells.hpp"
#include "landscape.hpp"
#include "many_simulations.hpp"
#include "random_pool.cuh" // NEW: Include the random pool header
#include "spread_functions.cuh"

#define DISTANCE 30
#define ELEVATION_MEAN 1163.3
#define ELEVATION_SD 399.5
#define UPPER_LIMIT 0.75

int main(int argc, char* argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <landscape_file_prefix>" << std::endl;
      return EXIT_FAILURE;
    }

    std::string landscape_file_prefix = argv[1];
    Landscape landscape(
        landscape_file_prefix + "-metadata.csv", landscape_file_prefix + "-landscape.csv"
    );

    IgnitionCells ignition_cells =
        read_ignition_cells(landscape_file_prefix + "-ignition_points.csv");

    SimulationParams params = { 0, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 };

    // --- NEW: Create the GPU random number resources before calling the simulation ---
    RandomPoolCuda random_pool(MAX_FRONTIER_SIZE, time(0));
    curandState* d_rand_states_ptr = random_pool.get_states_ptr();

    // --- CORRECTED: Pass the new pointer as the final argument ---
    Fire fire = simulate_fire(
        landscape, ignition_cells, params, DISTANCE, ELEVATION_MEAN, ELEVATION_SD, UPPER_LIMIT,
        d_rand_states_ptr // The required random state pointer
    );

    // Print the fire
    std::cout << "Landscape size: " << landscape.width << " " << landscape.height << std::endl;

    size_t step = 0;
    size_t i = 0;
    for (size_t j : fire.burned_ids_steps) {
      if (i >= j) {
        continue;
      }
      std::cout << "Step " << step << ":" << std::endl;
      for (; i < j; i++) {
        std::cout << fire.burned_ids[i].first << " " << fire.burned_ids[i].second << std::endl;
      }
      step++;
    }

  } catch (std::runtime_error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}