// filepath: spread_functions.cu

#include "spread_functions.cuh"

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(err)                                                                        \
  {                                                                                            \
    cudaError_t e = (err);                                                                     \
    if (e != cudaSuccess) {                                                                    \
      printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
      exit(EXIT_FAILURE);                                                                      \
    }                                                                                          \
  }

// --- CUDA Kernel and Device Functions (Unchanged) ---

__device__ float probability_func(
    const Cell& burning_cell, const Cell& neighbour_cell, float distance, const float angles[8],
    int neighbor_idx, const SimulationParams& params, float elevation_mean,
    float inv_elevation_sd, float upper_limit
) {
  if (distance == 0.0f)
    return 0.0f;
  float inv_dist = 1.0f / distance;

  float elev_diff = neighbour_cell.elevation - burning_cell.elevation;
  float slope_arg = elev_diff * inv_dist;
  float slope_term = slope_arg / sqrtf(1.0f + slope_arg * slope_arg);

  float wind_arg = angles[neighbor_idx] - burning_cell.wind_direction;
  float wind_term = cosf(wind_arg);

  float elev_term = (neighbour_cell.elevation - elevation_mean) * inv_elevation_sd;

  float linpred = params.independent_pred;
  if (neighbour_cell.vegetation_type == SUBALPINE)
    linpred += params.subalpine_pred;
  if (neighbour_cell.vegetation_type == WET)
    linpred += params.wet_pred;
  if (neighbour_cell.vegetation_type == DRY)
    linpred += params.dry_pred;

  linpred += params.fwi_pred * neighbour_cell.fwi;
  linpred += params.aspect_pred * neighbour_cell.aspect;
  linpred += params.wind_pred * wind_term;
  linpred += params.elevation_pred * elev_term;
  linpred += params.slope_pred * slope_term;

  float exp_term = expf(-linpred);
  return upper_limit / (1.0f + exp_term);
}

__global__ void spread_kernel(
    const uint2* d_frontier_in, size_t frontier_size, const Cell* d_landscape,
    int* d_burned_bin, uint2* d_frontier_out, int* d_frontier_out_count,
    curandState* d_rand_states, size_t n_col, size_t n_row, SimulationParams params,
    float distance, float elevation_mean, float inv_elevation_sd, float upper_limit
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= frontier_size)
    return;

  const int moves[2][8] = { { -1, -1, -1, 0, 0, 1, 1, 1 }, { -1, 0, 1, -1, 1, -1, 0, 1 } };
  const float angles[8] = { 3.1415926535f * 3.0f / 4.0f,
                            3.1415926535f,
                            3.1415926535f * 5.0f / 4.0f,
                            3.1415926535f / 2.0f,
                            3.1415926535f * 3.0f / 2.0f,
                            3.1415926535f / 4.0f,
                            0.0f,
                            3.1415926535f * 7.0f / 4.0f };

  uint2 burning_coord = d_frontier_in[idx];
  const Cell& burning_cell = d_landscape[burning_coord.y * n_col + burning_coord.x];
  curandState local_rand_state = d_rand_states[idx];

  for (int n = 0; n < 8; ++n) {
    int nx = burning_coord.x + moves[0][n];
    int ny = burning_coord.y + moves[1][n];

    if (nx < 0 || nx >= n_col || ny < 0 || ny >= n_row)
      continue;

    size_t neighbor_linear_idx = ny * n_col + nx;
    if (d_burned_bin[neighbor_linear_idx] == 1)
      continue;

    const Cell& neighbour_cell = d_landscape[neighbor_linear_idx];
    if (!neighbour_cell.burnable)
      continue;

    float prob = probability_func(
        burning_cell, neighbour_cell, distance, angles, n, params, elevation_mean,
        inv_elevation_sd, upper_limit
    );

    if (curand_uniform(&local_rand_state) < prob) {
      if (atomicExch(&d_burned_bin[neighbor_linear_idx], 1) == 0) {
        int out_idx = atomicAdd(d_frontier_out_count, 1);
        if (out_idx < MAX_FRONTIER_SIZE) {
          d_frontier_out[out_idx] = make_uint2(nx, ny);
        }
      }
    }
  }
  d_rand_states[idx] = local_rand_state;
}

// --- Host Function (Modified)---

Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit,
    curandState* d_rand_states_ptr // Use the provided pointer
) {
  size_t n_row = landscape.height;
  size_t n_col = landscape.width;
  size_t total_cells = n_row * n_col;

  // --- 1. Allocate Unified Memory ---
  Cell* d_landscape;
  int* d_burned_bin;
  uint2* d_frontier_in;
  uint2* d_frontier_out;
  int* d_frontier_out_count;

  CUDA_CHECK(cudaMallocManaged(&d_landscape, total_cells * sizeof(Cell)));
  CUDA_CHECK(cudaMallocManaged(&d_burned_bin, total_cells * sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&d_frontier_in, MAX_FRONTIER_SIZE * sizeof(uint2)));
  CUDA_CHECK(cudaMallocManaged(&d_frontier_out, MAX_FRONTIER_SIZE * sizeof(uint2)));
  CUDA_CHECK(cudaMallocManaged(&d_frontier_out_count, sizeof(int)));

  // --- 2. Initialize Data on the Host ---
  for (size_t i = 0; i < total_cells; ++i) {
    d_landscape[i] = landscape.cells.elems[i];
    d_burned_bin[i] = 0;
  }
  for (const auto& cell : ignition_cells) {
    if (cell.first < n_col && cell.second < n_row) {
      d_burned_bin[cell.second * n_col + cell.first] = 1;
    }
  }
  size_t current_step_burning_count = ignition_cells.size();
  if (current_step_burning_count > 0) {
    if (current_step_burning_count > MAX_FRONTIER_SIZE) {
      fprintf(
          stderr,
          "Error: Number of initial ignition points (%zu) exceeds MAX_FRONTIER_SIZE (%d).\n",
          current_step_burning_count, MAX_FRONTIER_SIZE
      );
      exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < current_step_burning_count; ++i) {
      d_frontier_in[i] = make_uint2(ignition_cells[i].first, ignition_cells[i].second);
    }
  }

  // --- 3. Prefetch Data to the GPU ---
  int deviceId;
  CUDA_CHECK(cudaGetDevice(&deviceId));
  CUDA_CHECK(cudaMemPrefetchAsync(d_landscape, total_cells * sizeof(Cell), deviceId, NULL));
  CUDA_CHECK(cudaMemPrefetchAsync(d_burned_bin, total_cells * sizeof(int), deviceId, NULL));
  if (current_step_burning_count > 0) {
    CUDA_CHECK(cudaMemPrefetchAsync(
        d_frontier_in, current_step_burning_count * sizeof(uint2), deviceId, NULL
    ));
  }

  // Random pool is no longer created here.
  std::vector<size_t> burned_ids_steps;
  burned_ids_steps.push_back(current_step_burning_count);

  // --- 4. Main Simulation Loop ---
  const int block_size = 256;
  float inv_elevation_sd = (elevation_sd != 0.0f) ? 1.0f / elevation_sd : 0.0f;

  while (current_step_burning_count > 0) {
    *d_frontier_out_count = 0;
    CUDA_CHECK(cudaMemPrefetchAsync(d_frontier_out_count, sizeof(int), deviceId, NULL));

    int num_blocks = (current_step_burning_count + block_size - 1) / block_size;

    spread_kernel<<<num_blocks, block_size>>>(
        d_frontier_in, current_step_burning_count, d_landscape, d_burned_bin, d_frontier_out,
        d_frontier_out_count, d_rand_states_ptr, n_col, n_row, params, distance, elevation_mean,
        inv_elevation_sd, upper_limit
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int new_cells_count = *d_frontier_out_count;

    if (new_cells_count == 0) {
      break;
    }

    std::swap(d_frontier_in, d_frontier_out);
    current_step_burning_count = new_cells_count;

    burned_ids_steps.push_back(burned_ids_steps.back() + new_cells_count);
  }

  // --- 5. Finalize and Clean Up ---
  Matrix<bool> final_burned_matrix(n_col, n_row);
  std::vector<std::pair<size_t, size_t>> burned_ids;
  burned_ids.reserve(burned_ids_steps.back());

  CUDA_CHECK(
      cudaMemPrefetchAsync(d_burned_bin, total_cells * sizeof(int), cudaCpuDeviceId, NULL)
  );
  CUDA_CHECK(cudaDeviceSynchronize());

  for (size_t j = 0; j < n_row; ++j) {
    for (size_t i = 0; i < n_col; ++i) {
      if (d_burned_bin[j * n_col + i] == 1) {
        final_burned_matrix.elems[j * n_col + i] = true;
        burned_ids.push_back({ i, j });
      } else {
        final_burned_matrix.elems[j * n_col + i] = false;
      }
    }
  }

  CUDA_CHECK(cudaFree(d_landscape));
  CUDA_CHECK(cudaFree(d_burned_bin));
  CUDA_CHECK(cudaFree(d_frontier_in));
  CUDA_CHECK(cudaFree(d_frontier_out));
  CUDA_CHECK(cudaFree(d_frontier_out_count));

  return { n_col, n_row, final_burned_matrix, burned_ids, burned_ids_steps };
}