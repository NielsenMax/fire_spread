#include "many_simulations.hpp"
#include <cmath>

Matrix<size_t> burned_amounts_per_cell(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit, size_t n_replicates
) {
  Matrix<size_t> burned_amounts(landscape.width, landscape.height);

  for (size_t i = 0; i < n_replicates; i++) {
    Fire fire = simulate_fire(
        landscape, ignition_cells, params, distance, elevation_mean, elevation_sd, upper_limit
    );

    // Iterate only over burned cells to improve efficiency
    for (const auto& cell : fire.burned_ids) {
      burned_amounts(cell.first, cell.second) += 1;
    }
  }

  return burned_amounts;
}
