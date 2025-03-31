#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

#include "fires.hpp"
#include "landscape.hpp"

/* minimax approximation to cos on [-pi/4, pi/4] with rel. err. ~= 7.5e-13 */
double cos_core(double x) {
  double x8, x4, x2;
  x2 = x * x;
  x4 = x2 * x2;
  x8 = x4 * x4;
  /* evaluate polynomial using Estrin's scheme */
  return (-2.7236370439787708e-7 * x2 + 2.4799852696610628e-5) * x8 +
         (-1.3888885054799695e-3 * x2 + 4.1666666636943683e-2) * x4 +
         (-4.9999999999963024e-1 * x2 + 1.0000000000000000e+0);
}

// Fast exp(-x) approximation (Schraudolph)
inline double fast_exp(double x) {
  return std::pow(2.0, 1.4426950408889634 * x);
}

double spread_probability(
    const Cell& burning, const Cell& neighbour, const SimulationParams& params, double angle,
    double distance, double elevation_mean, double inv_elevation_sd, double upper_limit = 1.0
) {
  // Precompute terms
  double elevation_diff = neighbour.elevation - burning.elevation;
  double slope_term = elevation_diff / distance;
  double wind_term = cos_core(angle - burning.wind_direction);
  double elev_term = (neighbour.elevation - elevation_mean) * inv_elevation_sd;

  // Base linear predictor
  double linpred = params.independent_pred;

  // Efficient vegetation type handling
  static const std::unordered_map<int, double> vegetation_map = {
    { SUBALPINE, params.subalpine_pred }, { WET, params.wet_pred }, { DRY, params.dry_pred }
  };

  if (auto it = vegetation_map.find(neighbour.vegetation_type); it != vegetation_map.end()) {
    linpred += it->second;
  }

  // Additive factors
  linpred += params.fwi_pred * neighbour.fwi;
  linpred += params.aspect_pred * neighbour.aspect;
  linpred += wind_term * params.wind_pred + elev_term * params.elevation_pred +
             slope_term * params.slope_pred;

  // Compute probability using fast exp
  return upper_limit / (1.0 + fast_exp(-linpred));
}

Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    const SimulationParams& params, double distance, double elevation_mean, double elevation_sd,
    double upper_limit = 1.0
) {
  const size_t n_row = landscape.height;
  const size_t n_col = landscape.width;

  constexpr double angles[8] = { M_PI * 3 / 4, M_PI, M_PI * 5 / 4, M_PI / 2, M_PI * 3 / 2,
                                 M_PI / 4,     0,    M_PI * 7 / 4 };
  constexpr int moves[8][2] = { { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 },
                                { 0, 1 },   { 1, -1 }, { 1, 0 },  { 1, 1 } };

  // Precompute 1/elevation_sd
  double inv_elevation_sd = 1.0 / elevation_sd;

  // Initialize burned cells tracking
  std::vector<std::pair<size_t, size_t>> burned_ids;
  burned_ids.reserve(n_row * n_col / 10);

  std::queue<std::pair<size_t, size_t>> fire_front;
  Matrix<bool> burned_bin(n_col, n_row);

  for (const auto& cell : ignition_cells) {
    burned_ids.push_back(cell);
    fire_front.push(cell);
    burned_bin(cell.first, cell.second) = true;
  }

  std::vector<size_t> burned_ids_steps = { ignition_cells.size() };

  // Static random number generator
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // Fire spread simulation using BFS-like queue
  while (!fire_front.empty()) {
    size_t level_size = fire_front.size();
    size_t new_burned = 0;

    for (size_t i = 0; i < level_size; ++i) {
      auto [burning_x, burning_y] = fire_front.front();
      fire_front.pop();
      const Cell& burning_cell = landscape[{ burning_x, burning_y }];

      for (size_t n = 0; n < 8; ++n) {
        int nx = burning_x + moves[n][0];
        int ny = burning_y + moves[n][1];

        // Skip out-of-bounds and already burned cells
        if (nx < 0 || ny < 0 || nx >= int(n_col) || ny >= int(n_row) || burned_bin(nx, ny))
          continue;

        const Cell& neighbor_cell = landscape[{ nx, ny }];
        if (!neighbor_cell.burnable)
          continue;

        double prob = spread_probability(
            burning_cell, neighbor_cell, params, angles[n], distance, elevation_mean,
            inv_elevation_sd, upper_limit
        );

        if (dist(gen) < prob) {
          burned_ids.emplace_back(nx, ny);
          fire_front.emplace(nx, ny);
          burned_bin(nx, ny) = true;
          new_burned++;
        }
      }
    }

    if (new_burned > 0)
      burned_ids_steps.push_back(burned_ids_steps.back() + new_burned);
  }

  return { n_col, n_row, burned_bin, burned_ids, burned_ids_steps };
}
