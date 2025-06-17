// filepath: many_simulations.cpp

#include "many_simulations.hpp"
#include "random_pool.cuh" // Include the random pool header here
#include <ctime>
#include <omp.h>
#include <vector>

// This definition must exist somewhere accessible, e.g. in spread_functions.cuh
#define MAX_FRONTIER_SIZE 8388608 // 2^23

Matrix<size_t> burned_amounts_per_cell(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit, size_t n_replicates
) {

  Matrix<size_t> burned_amounts(landscape.width, landscape.height);
  // Initialize burned_amounts to 0

  // --- OPTIMIZATION: Create one random pool per thread, ONCE, before the loop ---
  int max_threads = omp_get_max_threads();
  std::vector<RandomPoolCuda> thread_random_pools;
  thread_random_pools.reserve(max_threads);
  for (int i = 0; i < max_threads; ++i) {
    // emplace_back constructs the object in-place.
    // Give each pool a slightly different seed for better randomness.
    thread_random_pools.emplace_back(MAX_FRONTIER_SIZE, time(0) + i);
  }

// --- Run simulations in parallel ---
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < n_replicates; i++) {
    // Get the unique ID for the current OpenMP thread
    int thread_id = omp_get_thread_num();

    // Get the pointer to this thread's dedicated random state buffer
    curandState* d_rand_states_for_this_thread =
        thread_random_pools[thread_id].get_states_ptr();

    // Each thread simulates a fire independently, passing its own random state pointer
    Fire fire = simulate_fire(
        landscape, ignition_cells, params, distance, elevation_mean, elevation_sd, upper_limit,
        d_rand_states_for_this_thread // Pass the dedicated pointer
    );

    // Aggregate results atomically
    for (const auto& burned_cell_coords : fire.burned_ids) {
      if (burned_cell_coords.first < landscape.width &&
          burned_cell_coords.second < landscape.height) {
#pragma omp atomic update
        burned_amounts[burned_cell_coords] += 1;
      }
    }
  }

  return burned_amounts;
}