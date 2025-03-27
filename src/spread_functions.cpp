#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <vector>

#include "fires.hpp"
#include "landscape.hpp"

double spread_probability(
    const Cell& burning, const Cell& neighbour, SimulationParams params, double angle,
    double distance, double elevation_mean, double elevation_sd, double upper_limit = 1.0
) {

  double slope_term = sin(atan((neighbour.elevation - burning.elevation) / distance));
  double wind_term = cos(angle - burning.wind_direction);
  double elev_term = (neighbour.elevation - elevation_mean) / elevation_sd;

  double linpred = params.independent_pred;

  if (neighbour.vegetation_type == SUBALPINE) {
    linpred += params.subalpine_pred;
  } else if (neighbour.vegetation_type == WET) {
    linpred += params.wet_pred;
  } else if (neighbour.vegetation_type == DRY) {
    linpred += params.dry_pred;
  }

  linpred += params.fwi_pred * neighbour.fwi;
  linpred += params.aspect_pred * neighbour.aspect;

  linpred += wind_term * params.wind_pred + elev_term * params.elevation_pred +
             slope_term * params.slope_pred;

  double prob = upper_limit / (1 + exp(-linpred));

  return prob;
}

Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, double distance, double elevation_mean, double elevation_sd,
    double upper_limit = 1.0
) {
  const size_t n_row = landscape.height;
  const size_t n_col = landscape.width;

  // Pre-calculate angles for neighbor positions
  constexpr double angles[8] = { M_PI * 3 / 4, M_PI, M_PI * 5 / 4, M_PI / 2, M_PI * 3 / 2,
                                 M_PI / 4,     0,    M_PI * 7 / 4 };

  // Pre-calculate neighbor position offsets
  constexpr int moves[8][2] = { { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 },
                                { 0, 1 },   { 1, -1 }, { 1, 0 },  { 1, 1 } };

  // Initialize burned cells tracking
  std::vector<std::pair<size_t, size_t>> burned_ids(ignition_cells);
  std::vector<size_t> burned_ids_steps = { ignition_cells.size() };

  // Initialize burned cells matrix
  Matrix<bool> burned_bin(n_col, n_row);
  for (const auto& cell : ignition_cells) {
    burned_bin[{ cell.first, cell.second }] = true;
  }

  // Initialize random number generator for better performance
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // Fire spread simulation
  size_t start = 0;
  size_t end = ignition_cells.size();

  while (end > start) { // Only continue if there are new burning cells
    size_t end_forward = end;

    // Process all cells that started burning in the previous iteration
    for (size_t b = start; b < end; b++) {
      const size_t burning_cell_0 = burned_ids[b].first;
      const size_t burning_cell_1 = burned_ids[b].second;
      const Cell& burning_cell = landscape[{ burning_cell_0, burning_cell_1 }];

      // Check all 8 neighboring cells
      for (size_t n = 0; n < 8; n++) {
        const int neighbour_cell_0 = int(burning_cell_0) + moves[n][0];
        const int neighbour_cell_1 = int(burning_cell_1) + moves[n][1];

        // Skip if out of bounds
        if (neighbour_cell_0 < 0 || neighbour_cell_0 >= int(n_col) || neighbour_cell_1 < 0 ||
            neighbour_cell_1 >= int(n_row)) {
          continue;
        }

        // Skip if already burned or not burnable
        if (burned_bin[{ neighbour_cell_0, neighbour_cell_1 }]) {
          continue;
        }

        const Cell& neighbour_cell = landscape[{ neighbour_cell_0, neighbour_cell_1 }];
        if (!neighbour_cell.burnable) {
          continue;
        }

        // Calculate spread probability
        const double prob = spread_probability(
            burning_cell, neighbour_cell, params, angles[n], distance, elevation_mean,
            elevation_sd, upper_limit
        );

        // Determine if fire spreads to this cell
        if (dist(gen) < prob) {
          burned_ids.emplace_back(neighbour_cell_0, neighbour_cell_1);
          burned_bin[{ neighbour_cell_0, neighbour_cell_1 }] = true;
          end_forward++;
        }
      }
    }

    // Update indices for next iteration
    start = end;
    end = end_forward;
    burned_ids_steps.push_back(end);
  }

  return { n_col, n_row, burned_bin, burned_ids, burned_ids_steps };
}
