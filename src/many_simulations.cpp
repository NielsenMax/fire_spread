#include "many_simulations.hpp"

#include <cmath>
#include <omp.h> // Include OpenMP header

Matrix<size_t> burned_amounts_per_cell(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit, size_t n_replicates
) {

  Matrix<size_t> burned_amounts(landscape.width, landscape.height);
  // Initialize burned_amounts to 0, std::vector constructor does this for size_t.

  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < n_replicates; i++) {
    // Each thread simulates a fire independently
    Fire fire = simulate_fire(
        landscape, ignition_cells, params, distance, elevation_mean, elevation_sd, upper_limit
    );

    // Iterate over the cells burned in this specific fire simulation
    // This loop is part of the parallel replicate, so updates to shared 'burned_amounts' must be atomic.
    for (const auto& burned_cell_coords : fire.burned_ids) {
        // fire.burned_ids contains pairs {col, row}
        // Ensure coordinates are valid before attempting atomic update, though simulate_fire should ensure this.
        if (burned_cell_coords.first < landscape.width && burned_cell_coords.second < landscape.height) {
            #pragma omp atomic update
            burned_amounts[burned_cell_coords] += 1;
        }
    }
    // The original nested loop iterating through the entire landscape can be less efficient
    // if the number of burned cells is much smaller than the total landscape size.
    // Using fire.burned_ids is generally better.
  }

  return burned_amounts;
}
