// filepath: spread_functions.cuh

#pragma once

#include "fires.hpp"
#include "landscape.hpp"
#include <utility>
#include <vector>

// CORRECTED: Include the full header to get the one true definition of curandState.
// This replaces the line "struct curandState;".
#include <curand_kernel.h>

#define MAX_FRONTIER_SIZE 8388608

struct SimulationParams {
  float independent_pred;
  float wind_pred;
  float elevation_pred;
  float slope_pred;
  float subalpine_pred;
  float wet_pred;
  float dry_pred;
  float fwi_pred;
  float aspect_pred;
};

Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit, curandState* d_rand_states_for_this_thread
);