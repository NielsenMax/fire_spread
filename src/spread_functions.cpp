// spread_functions.cpp (With Explicit AVX2 Vectorization for probability calculation)

#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>     // For uint8_t
#include <immintrin.h> // For AVX2 intrinsics
#include <queue>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fires.hpp"
#include "landscape.hpp" // Uses the SoA landscape.hpp
#include "matrix.hpp"

// --- AVX2 Helper Functions ---

// Vectorized cos_core using polynomial approximation and FMA
inline __m256 cos_core_vec(__m256 x) {
  const __m256 c2_term = _mm256_set1_ps(-2.7236370439787708e-7f);
  const __m256 c0_term = _mm256_set1_ps(2.4799852696610628e-5f);
  const __m256 d2_term = _mm256_set1_ps(-1.3888885054799695e-3f);
  const __m256 d0_term = _mm256_set1_ps(4.1666666636943683e-2f);
  const __m256 e2_term = _mm256_set1_ps(-4.9999999999963024e-1f);
  const __m256 e0_term = _mm256_set1_ps(1.0000000000000000e+0f);
  __m256 x2 = _mm256_mul_ps(x, x);
  __m256 x4 = _mm256_mul_ps(x2, x2);
  __m256 x8 = _mm256_mul_ps(x4, x4);
  __m256 term1 = _mm256_fmadd_ps(c2_term, x2, c0_term);
  __m256 term2 = _mm256_fmadd_ps(d2_term, x2, d0_term);
  __m256 term3 = _mm256_fmadd_ps(e2_term, x2, e0_term);
  return _mm256_fmadd_ps(term1, x8, _mm256_fmadd_ps(term2, x4, term3));
}

// Vectorized fast_exp_neg_poly using polynomial approximation and FMA
inline __m256 fast_exp_neg_poly_vec(__m256 x) {
  const __m256 neg_log2_e = _mm256_set1_ps(-1.4426950408889634f);
  const __m256 y_max = _mm256_set1_ps(10.0f);
  const __m256 y_min = _mm256_set1_ps(-10.0f);
  __m256 y = _mm256_mul_ps(x, neg_log2_e);
  y = _mm256_max_ps(y_min, _mm256_min_ps(y_max, y));
  const __m256 c0 = _mm256_set1_ps(1.0f);
  const __m256 c1 = _mm256_set1_ps(0.69315307f);
  const __m256 c2 = _mm256_set1_ps(0.24015361f);
  const __m256 c3 = _mm256_set1_ps(0.055826318f);
  const __m256 c4 = _mm256_set1_ps(0.0096318756f);
  const __m256 c5 = _mm256_set1_ps(0.0013391345f);
  __m256 result = _mm256_fmadd_ps(c5, y, c4);
  result = _mm256_fmadd_ps(result, y, c3);
  result = _mm256_fmadd_ps(result, y, c2);
  result = _mm256_fmadd_ps(result, y, c1);
  result = _mm256_fmadd_ps(result, y, c0);
  return result;
}
// --- End AVX2 Helper Functions ---

// --- Constants (Using original __attribute__ as it worked) ---
constexpr float angles[8] __attribute__((aligned(32))
) = { M_PI * 3 / 4, M_PI, M_PI * 5 / 4, M_PI / 2, M_PI * 3 / 2, M_PI / 4, 0, M_PI * 7 / 4 };
constexpr int moves_flat[16]
    __attribute__((aligned(32))) = { -1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1 };
// --- End Constants ---

// --- Random Number Generation (Use bitwise modulo) ---
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
class RandomPool {
private:
  static constexpr size_t POOL_SIZE = 1024;
  static_assert(
      (POOL_SIZE > 0) && ((POOL_SIZE & (POOL_SIZE - 1)) == 0), "POOL_SIZE must be a power of 2"
  );
  std::vector<float> random_values;
  size_t current_index;

public:
  RandomPool() : random_values(POOL_SIZE), current_index(0) {
    for (size_t i = 0; i < POOL_SIZE; ++i)
      random_values[i] = dist(gen);
  }
  inline float get() {
    float v = random_values[current_index];
    current_index = (current_index + 1) & (POOL_SIZE - 1);
    return v;
  }
  inline void get_batch(float* ob, size_t c) {
    for (size_t i = 0; i < c; ++i)
      ob[i] = get();
  }
  void refill() {
    if (current_index > POOL_SIZE / 2) {
      for (size_t i = 0; i < POOL_SIZE; ++i)
        random_values[i] = dist(gen);
      current_index = 0;
    }
  }
};
static RandomPool random_pool;
// --- End Random Number Generation ---

// --- Spread Probability Calculation (Explicit AVX2 Intrinsics) ---
// Replaces the #pragma omp simd version
void spread_probability_vectorized_intrinsics(
    float burning_elevation_param, float burning_wind_direction_param,
    const float* neighbor_elevations, const uint8_t* neighbor_veg_types_uint8,
    const float* neighbor_fwis, const float* neighbor_aspects, const SimulationParams& params,
    const float* angles_arg, float distance, float elevation_mean, float inv_elevation_sd,
    float* probabilities,   // Output
    const bool* valid_mask, // Still need mask for conditional store/blend
    int n, float upper_limit = 1.0
) {
  // Constants as vectors
  const __m256 vec_upper_limit = _mm256_set1_ps(upper_limit);
  const __m256 vec_one = _mm256_set1_ps(1.0f);
  const __m256 vec_zero = _mm256_setzero_ps();
  const __m256 vec_burning_elevation = _mm256_set1_ps(burning_elevation_param);
  const __m256 vec_burning_wind_dir = _mm256_set1_ps(burning_wind_direction_param);
  const __m256 vec_elevation_mean = _mm256_set1_ps(elevation_mean);
  const __m256 vec_inv_elevation_sd = _mm256_set1_ps(inv_elevation_sd);
  const __m256 vec_distance = _mm256_set1_ps(distance);
  const __m256 vec_independent_pred = _mm256_set1_ps(params.independent_pred);
  const __m256 vec_subalpine_pred = _mm256_set1_ps(params.subalpine_pred);
  const __m256 vec_wet_pred = _mm256_set1_ps(params.wet_pred);
  const __m256 vec_dry_pred = _mm256_set1_ps(params.dry_pred);
  const __m256 vec_matorral_pred = _mm256_setzero_ps(); // Assuming 0 for Matorral
  const __m256 vec_fwi_pred = _mm256_set1_ps(params.fwi_pred);
  const __m256 vec_aspect_pred = _mm256_set1_ps(params.aspect_pred);
  const __m256 vec_wind_pred = _mm256_set1_ps(params.wind_pred);
  const __m256 vec_elevation_pred = _mm256_set1_ps(params.elevation_pred);
  const __m256 vec_slope_pred = _mm256_set1_ps(params.slope_pred);
  const __m256 vec_inv_distance =
      (distance != 0.0f) ? _mm256_div_ps(vec_one, vec_distance) : vec_zero;

  assert(n == 8 && "This intrinsic version assumes n=8");

  // Load validity mask (bool[8] -> int[8] -> __m256i -> __m256)
  alignas(32) int valid_int_mask[8];
  for (int i = 0; i < 8; ++i)
    valid_int_mask[i] = valid_mask[i] ? -1 : 0;
  __m256i vec_valid_mask_i = _mm256_load_si256((const __m256i*)valid_int_mask);
  __m256 vec_valid_mask = _mm256_castsi256_ps(vec_valid_mask_i);

  // Load neighbor data (use aligned loads as source uses __attribute__)
  __m256 vec_neighbor_elev = _mm256_load_ps(neighbor_elevations);
  __m256 vec_neighbor_fwi = _mm256_load_ps(neighbor_fwis);
  __m256 vec_neighbor_aspect = _mm256_load_ps(neighbor_aspects);
  __m256 vec_angles = _mm256_load_ps(angles_arg);
  __m128i veg_types_low = _mm_loadl_epi64((const __m128i*)neighbor_veg_types_uint8);
  __m256i vec_veg_types_i = _mm256_cvtepu8_epi32(veg_types_low);

  // --- Start Calculations ---
  __m256 vec_elev_diff = _mm256_sub_ps(vec_neighbor_elev, vec_burning_elevation);
  __m256 vec_slope_term = _mm256_mul_ps(vec_elev_diff, vec_inv_distance);
  __m256 vec_angle_diff = _mm256_sub_ps(vec_angles, vec_burning_wind_dir);
  __m256 vec_wind_term = cos_core_vec(vec_angle_diff);
  __m256 vec_elev_term_tmp = _mm256_sub_ps(vec_neighbor_elev, vec_elevation_mean);
  __m256 vec_elev_term = _mm256_mul_ps(vec_elev_term_tmp, vec_inv_elevation_sd);
  __m256 vec_linpred = vec_independent_pred;

  // Vegetation Type Term (blends)
  __m256i vec_veg_subalpine = _mm256_set1_epi32(SUBALPINE);
  __m256i vec_veg_wet = _mm256_set1_epi32(WET);
  __m256i vec_veg_dry = _mm256_set1_epi32(DRY);
  __m256i mask_subalpine = _mm256_cmpeq_epi32(vec_veg_types_i, vec_veg_subalpine);
  __m256i mask_wet = _mm256_cmpeq_epi32(vec_veg_types_i, vec_veg_wet);
  __m256i mask_dry = _mm256_cmpeq_epi32(vec_veg_types_i, vec_veg_dry);
  __m256 mask_subalpine_f = _mm256_castsi256_ps(mask_subalpine);
  __m256 mask_wet_f = _mm256_castsi256_ps(mask_wet);
  __m256 mask_dry_f = _mm256_castsi256_ps(mask_dry);
  __m256 veg_term = vec_matorral_pred; // Start with default (0)
  veg_term = _mm256_blendv_ps(veg_term, vec_subalpine_pred, mask_subalpine_f);
  veg_term = _mm256_blendv_ps(veg_term, vec_wet_pred, mask_wet_f);
  veg_term = _mm256_blendv_ps(veg_term, vec_dry_pred, mask_dry_f);
  vec_linpred = _mm256_add_ps(vec_linpred, veg_term);

  // Add other terms using FMA
  vec_linpred = _mm256_fmadd_ps(vec_fwi_pred, vec_neighbor_fwi, vec_linpred);
  vec_linpred = _mm256_fmadd_ps(vec_aspect_pred, vec_neighbor_aspect, vec_linpred);
  vec_linpred = _mm256_fmadd_ps(vec_wind_term, vec_wind_pred, vec_linpred);
  vec_linpred = _mm256_fmadd_ps(vec_elev_term, vec_elevation_pred, vec_linpred);
  vec_linpred = _mm256_fmadd_ps(vec_slope_term, vec_slope_pred, vec_linpred);

  // Calculate probability
  __m256 vec_exp_term = fast_exp_neg_poly_vec(vec_linpred);
  __m256 vec_denominator = _mm256_add_ps(vec_one, vec_exp_term);
  __m256 vec_prob_raw = _mm256_div_ps(vec_upper_limit, vec_denominator);

  // Apply validity mask
  __m256 vec_prob_final = _mm256_and_ps(vec_prob_raw, vec_valid_mask);

  // Store results (use aligned store as probabilities should be aligned in neighbor_data)
  _mm256_store_ps(probabilities, vec_prob_final);
}
// --- End Spread Probability ---

// --- Burned Cells Tracker (Unchanged) ---
struct BurnedCellsTracker {
  Matrix<bool> burned_bin;
  std::vector<std::pair<size_t, size_t>> burned_ids;
  std::vector<size_t> burned_ids_steps;
  BurnedCellsTracker(size_t w, size_t h) : burned_bin(w, h) {
    burned_ids.reserve(w * h / 10);
    burned_ids_steps.push_back(0);
  }
  void add_cell(size_t x, size_t y) {
    if (!burned_bin(x, y)) {
      burned_ids.emplace_back(x, y);
      burned_bin(x, y) = true;
    }
  }
  void update_steps(size_t n) {
    if (n > 0)
      burned_ids_steps.push_back(burned_ids_steps.back() + n);
  }
  bool is_burned(size_t x, size_t y) const {
    return burned_bin(x, y);
  }
};
// --- End Burned Cells Tracker ---

// --- Main Simulation Function (Updated to call intrinsic version) ---
Fire simulate_fire(
    const Landscape& landscape, // SoA Landscape
    const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    const SimulationParams& params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit = 1.0
) {
  const size_t n_row = landscape.height;
  const size_t n_col = landscape.width;
  const float inv_elevation_sd = (elevation_sd != 0.0f) ? (1.0f / elevation_sd) : 0.0f;
  BurnedCellsTracker burned_tracker(n_col, n_row);
  std::vector<std::pair<size_t, size_t>> current_level, next_level;
  current_level.reserve(n_row * n_col / 10);
  next_level.reserve(n_row * n_col / 10);
  // Initialization (Unchanged)
  std::vector<bool> ignition_added(n_col * n_row, false);
  for (const auto& cc : ignition_cells) {
    if (cc.first < n_col && cc.second < n_row) {
      size_t idx = cc.second * n_col + cc.first;
      if (landscape.is_burnable(cc.first, cc.second) && !ignition_added[idx]) {
        burned_tracker.add_cell(cc.first, cc.second);
        current_level.push_back(cc);
        ignition_added[idx] = true;
      }
    }
  }

  // Neighbor Data Struct (SoA - unchanged)
  struct AlignedNeighborData {
    float elevations[8] __attribute__((aligned(32)));
    uint8_t veg_types_uint8[8] __attribute__((aligned(32))); // Store underlying type
    float fwis[8] __attribute__((aligned(32)));
    float aspects[8] __attribute__((aligned(32)));
    float angles[8] __attribute__((aligned(32)));
    bool valid[8] __attribute__((aligned(32)));
    float probs[8] __attribute__((aligned(32)));
    int x[8] __attribute__((aligned(32)));
    int y[8] __attribute__((aligned(32)));
  } neighbor_data;

  // Main loop
  while (!current_level.empty()) {
    next_level.clear();
    size_t new_burned = 0;

    for (const auto& cell_coords : current_level) {
      const size_t cell_x = cell_coords.first;
      const size_t cell_y = cell_coords.second;
      const float burning_elevation = landscape.get_elevation(cell_x, cell_y);
      const float burning_wind_direction = landscape.get_wind_direction(cell_x, cell_y);

      // Prefetching (Unchanged)
      if (&cell_coords != &current_level.back()) {
        const auto& next_cell_coords = *(&cell_coords + 1);
        const size_t next_x = next_cell_coords.first;
        const size_t next_y = next_cell_coords.second;
        if (next_x < n_col && next_y < n_row) {
          size_t next_linear_idx = landscape.burnable_m.index(next_x, next_y);
          size_t next_byte_idx = landscape.burnable_m.byte_index(next_linear_idx);
          if (next_byte_idx < landscape.burnable_m.elems.size()) {
            __builtin_prefetch(&landscape.burnable_m.elems[next_byte_idx], 0, 0);
          }
          __builtin_prefetch(&landscape.elevations(next_x, next_y), 0, 0);
        }
      }

      // Vectorized Neighbor Coordinate Calculation & Bounds Check (Unchanged)
      __m256i cell_x_vec = _mm256_set1_epi32(static_cast<int>(cell_x));
      __m256i cell_y_vec = _mm256_set1_epi32(static_cast<int>(cell_y));
      __m256i moves_x_vec = _mm256_set_epi32(
          moves_flat[14], moves_flat[12], moves_flat[10], moves_flat[8], moves_flat[6],
          moves_flat[4], moves_flat[2], moves_flat[0]
      );
      __m256i moves_y_vec = _mm256_set_epi32(
          moves_flat[15], moves_flat[13], moves_flat[11], moves_flat[9], moves_flat[7],
          moves_flat[5], moves_flat[3], moves_flat[1]
      );
      __m256i neighbor_x_vec = _mm256_add_epi32(cell_x_vec, moves_x_vec);
      __m256i neighbor_y_vec = _mm256_add_epi32(cell_y_vec, moves_y_vec);
      _mm256_store_si256((__m256i*)neighbor_data.x, neighbor_x_vec);
      _mm256_store_si256((__m256i*)neighbor_data.y, neighbor_y_vec);
      __m256i x_ge_zero = _mm256_cmpgt_epi32(neighbor_x_vec, _mm256_set1_epi32(-1));
      __m256i y_ge_zero = _mm256_cmpgt_epi32(neighbor_y_vec, _mm256_set1_epi32(-1));
      __m256i n_col_vec = _mm256_set1_epi32(static_cast<int>(n_col));
      __m256i n_row_vec = _mm256_set1_epi32(static_cast<int>(n_row));
      __m256i x_lt_ncol = _mm256_cmpgt_epi32(n_col_vec, neighbor_x_vec);
      __m256i y_lt_nrow = _mm256_cmpgt_epi32(n_row_vec, neighbor_y_vec);
      __m256i bounds_mask = _mm256_and_si256(
          _mm256_and_si256(x_ge_zero, y_ge_zero), _mm256_and_si256(x_lt_ncol, y_lt_nrow)
      );

      // Neighbor Validation Loop (Unchanged - populates neighbor_data SoA)
      int bounds_check_array[8] __attribute__((aligned(32)));
      _mm256_store_si256((__m256i*)bounds_check_array, bounds_mask);
      for (int n = 0; n < 8; ++n) {
        neighbor_data.valid[n] = false;
      }
      int valid_neighbor_count = 0;
      for (int batch = 0; batch < 8; batch += 4) {
        // Prefetching next batch (Unchanged)
        if (batch + 4 < 8) { /* ... */
        }
        // Process current batch (Unchanged)
        for (int n = batch; n < std::min(batch + 4, 8); ++n) {
          if (bounds_check_array[n]) {
            const int nx = neighbor_data.x[n];
            const int ny = neighbor_data.y[n];
            const size_t snx = static_cast<size_t>(nx);
            const size_t sny = static_cast<size_t>(ny);
            if (!burned_tracker.is_burned(snx, sny) && landscape.is_burnable(snx, sny)) {
              neighbor_data.valid[n] = true;
              neighbor_data.elevations[n] = landscape.get_elevation(snx, sny);
              neighbor_data.veg_types_uint8[n] =
                  static_cast<uint8_t>(landscape.get_vegetation_type(snx, sny));
              neighbor_data.fwis[n] = landscape.get_fwi(snx, sny);
              neighbor_data.aspects[n] = landscape.get_aspect(snx, sny);
              neighbor_data.angles[n] = angles[n];
              valid_neighbor_count++;
            }
          }
        }
      } // End batch loop

      // Calculate Probabilities (Call *** INTRINSIC *** version)
      if (valid_neighbor_count > 0) {
        spread_probability_vectorized_intrinsics( // *** CALL NEW VERSION ***
            burning_elevation, burning_wind_direction, neighbor_data.elevations,
            neighbor_data.veg_types_uint8, neighbor_data.fwis, neighbor_data.aspects, params,
            neighbor_data.angles, distance, elevation_mean, inv_elevation_sd,
            neighbor_data.probs, neighbor_data.valid, 8, upper_limit
        );
      } else {
        continue;
      }

      // Vectorized Result Processing (Unchanged)
      alignas(32) float rand_buffer[8];
      random_pool.get_batch(rand_buffer, 8);
      __m256 probs_vec = _mm256_loadu_ps(neighbor_data.probs);
      __m256 rand_vec = _mm256_load_ps(rand_buffer);
      __m256 cmp_mask_ps = _mm256_cmp_ps(rand_vec, probs_vec, _CMP_LT_OQ);
      __m256i cmp_mask_i = _mm256_castps_si256(cmp_mask_ps);
      alignas(32) int valid_mask_array[8];
      for (int n = 0; n < 8; ++n) {
        valid_mask_array[n] = neighbor_data.valid[n] ? -1 : 0;
      }
      __m256i valid_mask_i = _mm256_loadu_si256((const __m256i*)valid_mask_array);
      __m256i final_burn_mask_i = _mm256_and_si256(valid_mask_i, cmp_mask_i);
      alignas(32) int final_burn_mask_array[8];
      _mm256_store_si256((__m256i*)final_burn_mask_array, final_burn_mask_i);
      for (int n = 0; n < 8; ++n) {
        if (final_burn_mask_array[n]) {
          const int nx = neighbor_data.x[n];
          const int ny = neighbor_data.y[n];
          const size_t snx = static_cast<size_t>(nx);
          const size_t sny = static_cast<size_t>(ny);
          if (!burned_tracker.is_burned(snx, sny)) {
            next_level.emplace_back(snx, sny);
            burned_tracker.add_cell(snx, sny);
            new_burned++;
          }
        }
      }

    } // End loop over current_level

    burned_tracker.update_steps(new_burned);
    random_pool.refill();
    std::swap(current_level, next_level);
  } // End while loop

  return { n_col, n_row, std::move(burned_tracker.burned_bin),
           std::move(burned_tracker.burned_ids), std::move(burned_tracker.burned_ids_steps) };
}