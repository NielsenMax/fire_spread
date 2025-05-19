// spread_functions.cpp  (Hybrid BFS / full-grid, OpenMP-parallelized)

#include "spread_functions.hpp"

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <omp.h>
#include <random>
#include <utility>
#include <vector>

#include "fires.hpp"
#include "landscape.hpp" // SoA Landscape
#include "matrix.hpp"    // Matrix<T>, including Matrix<bool>

// --- AVX2 Helper Functions ---
inline __m256 cos_core_vec(__m256 x) {
  const __m256 c2 = _mm256_set1_ps(-2.7236370439787708e-7f);
  const __m256 c0 = _mm256_set1_ps(2.4799852696610628e-5f);
  const __m256 d2 = _mm256_set1_ps(-1.3888885054799695e-3f);
  const __m256 d0 = _mm256_set1_ps(4.1666666636943683e-2f);
  const __m256 e2 = _mm256_set1_ps(-4.9999999999963024e-1f);
  const __m256 e0 = _mm256_set1_ps(1.0000000000000000e+0f);
  __m256 x2 = _mm256_mul_ps(x, x);
  __m256 x4 = _mm256_mul_ps(x2, x2);
  __m256 x8 = _mm256_mul_ps(x4, x4);
  __m256 t1 = _mm256_fmadd_ps(c2, x2, c0);
  __m256 t2 = _mm256_fmadd_ps(d2, x2, d0);
  __m256 t3 = _mm256_fmadd_ps(e2, x2, e0);
  return _mm256_fmadd_ps(t1, x8, _mm256_fmadd_ps(t2, x4, t3));
}

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
  __m256 r = _mm256_fmadd_ps(c5, y, c4);
  r = _mm256_fmadd_ps(r, y, c3);
  r = _mm256_fmadd_ps(r, y, c2);
  r = _mm256_fmadd_ps(r, y, c1);
  return _mm256_fmadd_ps(r, y, c0);
}

// --- Constants ---
constexpr float angles[8] __attribute__((aligned(32))
) = { M_PI * 3 / 4, M_PI, M_PI * 5 / 4, M_PI / 2, M_PI * 3 / 2, M_PI / 4, 0.0f, M_PI * 7 / 4 };
constexpr int moves_flat[16]
    __attribute__((aligned(32))) = { -1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1 };

// --- RandomPool (one per thread) ---
class RandomPool {
  static constexpr size_t POOL_SIZE = 1024;
  std::vector<float> rnd;
  size_t idx = 0;

public:
  RandomPool() : rnd(POOL_SIZE), idx(0) {
    std::mt19937 gen{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : rnd)
      v = dist(gen);
  }
  inline float get() {
    float v = rnd[idx];
    idx = (idx + 1) & (POOL_SIZE - 1);
    return v;
  }
  inline void get_batch(float* out, size_t n) {
    for (size_t i = 0; i < n; ++i)
      out[i] = get();
  }
  void refill() {
    if (idx > POOL_SIZE / 2) {
      std::mt19937 gen{ std::random_device{}() };
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (auto& v : rnd)
        v = dist(gen);
      idx = 0;
    }
  }
};
static thread_local RandomPool random_pool;

// --- BurnedCellsTracker (flat, atomic) ---
struct BurnedCellsTracker {
  size_t width, height;
  std::vector<char> burned_bin; // 0/1 flags
  std::vector<std::pair<size_t, size_t>> burned_ids;
  std::vector<size_t> burned_ids_steps;

  BurnedCellsTracker(size_t w, size_t h) : width(w), height(h), burned_bin(w * h, 0) {
    burned_ids.reserve(w * h / 10);
    burned_ids_steps.push_back(0);
  }

  void add_cell_seq(size_t x, size_t y) {
    size_t i = y * width + x;
    if (!burned_bin[i]) {
      burned_bin[i] = 1;
      burned_ids.emplace_back(x, y);
    }
  }

  bool try_mark_burn(size_t x, size_t y) {
    size_t i = y * width + x;
    char expected = 0;
    return __atomic_compare_exchange_n(
        &burned_bin[i], &expected, char(1),
        /*weak=*/false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
    );
  }
};

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

// --- Main Simulation Function ---
Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    const SimulationParams& params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit /*=1.0f*/
) {
  const size_t n_row = landscape.height;
  const size_t n_col = landscape.width;
  const float inv_elev_sd = elevation_sd > 0.0f ? 1.0f / elevation_sd : 0.0f;

  BurnedCellsTracker burned_tracker(n_col, n_row);

  // --- initialize ignition level (single-threaded) ---
  std::vector<std::pair<size_t, size_t>> current_level;
  current_level.reserve(n_row * n_col / 10);
  std::vector<char> ignition_added(n_col * n_row, 0);

  for (auto [x, y] : ignition_cells) {
    if (x < n_col && y < n_row && landscape.is_burnable(x, y)) {
      size_t idx = y * n_col + x;
      if (!ignition_added[idx]) {
        burned_tracker.add_cell_seq(x, y);
        current_level.emplace_back(x, y);
        ignition_added[idx] = 1;
      }
    }
  }

  // --- time‐step loop ---
  while (!current_level.empty()) {
    std::vector<std::pair<size_t, size_t>> next_level;
    next_level.reserve(current_level.size());
    size_t new_burned = 0;

#pragma omp parallel
    {
      // per-thread neighbor SoA
      struct Neigh {
        alignas(32) float elev[8];
        uint8_t veg[8]; // Alignment for this is less critical for _mm_loadl_epi64
        alignas(32) float fwi[8];
        alignas(32) float asp[8];
        alignas(32) float ang[8];
        bool val[8]; // Not directly loaded into __m256 with aligned load
        alignas(32) float prob[8];
        // int x[8], y[8] are stored with _mm256_storeu_si256 (unaligned is fine)
        // but for consistency or if other aligned ops were used, could be aligned too.
        // For now, focusing on the ones causing GPF with _load_ps/_store_ps.
        int x[8];
        int y[8];
      } nd;

      std::vector<std::pair<size_t, size_t>> local_next, local_burned;
      local_next.reserve(64);
      local_burned.reserve(64);

#pragma omp for schedule(dynamic) reduction(+ : new_burned)
      for (int ic = 0; ic < (int)current_level.size(); ++ic) {
        auto [cx, cy] = current_level[ic];
        float belev = landscape.get_elevation(cx, cy);
        float bwind = landscape.get_wind_direction(cx, cy);

        // --- compute neighbor coords & bounds (same as original) ---
        __m256i cxv = _mm256_set1_epi32((int)cx);
        __m256i cyv = _mm256_set1_epi32((int)cy);
        __m256i mx = _mm256_set_epi32(
            moves_flat[14], moves_flat[12], moves_flat[10], moves_flat[8], moves_flat[6],
            moves_flat[4], moves_flat[2], moves_flat[0]
        );
        __m256i my = _mm256_set_epi32(
            moves_flat[15], moves_flat[13], moves_flat[11], moves_flat[9], moves_flat[7],
            moves_flat[5], moves_flat[3], moves_flat[1]
        );
        __m256i nxv = _mm256_add_epi32(cxv, mx);
        __m256i nyv = _mm256_add_epi32(cyv, my);

        _mm256_storeu_si256((__m256i*)nd.x, nxv);
        _mm256_storeu_si256((__m256i*)nd.y, nyv);

        __m256i ge0x = _mm256_cmpgt_epi32(nxv, _mm256_set1_epi32(-1));
        __m256i ge0y = _mm256_cmpgt_epi32(nyv, _mm256_set1_epi32(-1));
        __m256i ltX = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)n_col), nxv);
        __m256i ltY = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)n_row), nyv);
        __m256i bmask =
            _mm256_and_si256(_mm256_and_si256(ge0x, ge0y), _mm256_and_si256(ltX, ltY));
        int bchk[8];
        _mm256_storeu_si256((__m256i*)bchk, bmask);

        // --- fill nd.val[] & SoA if burnable & not yet burned ---
        int valid_count = 0;
        for (int b = 0; b < 8; ++b)
          nd.val[b] = false;
        for (int b = 0; b < 8; ++b) {
          if (!bchk[b])
            continue;
          size_t nx = (size_t)nd.x[b], ny = (size_t)nd.y[b];
          // only claim if the cell IS burnable; `try_mark_burn` flips 0→1 on first call
          if (!landscape.is_burnable(nx, ny))
            continue;
          // we _don't_ want to mark it permanently here yet; we'll flip later if RNG decides
          nd.val[b] = true;
          nd.elev[b] = landscape.get_elevation(nx, ny);
          nd.veg[b] = (uint8_t)landscape.get_vegetation_type(nx, ny);
          nd.fwi[b] = landscape.get_fwi(nx, ny);
          nd.asp[b] = landscape.get_aspect(nx, ny);
          nd.ang[b] = angles[b];
          ++valid_count;
        }
        if (!valid_count)
          continue;

        // --- compute probabilities via AVX2 intrinsics ---
        spread_probability_vectorized_intrinsics(
            belev, bwind, nd.elev, nd.veg, nd.fwi, nd.asp, params, nd.ang, distance,
            elevation_mean, inv_elev_sd, nd.prob, nd.val, 8, upper_limit
        );

        // --- compare to randoms & decide burns ---
        float rnds[8];
        random_pool.get_batch(rnds, 8);
        __m256 probs_v = _mm256_loadu_ps(nd.prob);
        __m256 rnd_v = _mm256_loadu_ps(rnds);
        __m256 cmp_v = _mm256_cmp_ps(rnd_v, probs_v, _CMP_LT_OQ);
        __m256i cmp_i = _mm256_castps_si256(cmp_v);

        int maskv[8];
        for (int b = 0; b < 8; ++b)
          maskv[b] = nd.val[b] ? -1 : 0;
        __m256i val_i = _mm256_loadu_si256((__m256i*)maskv);
        __m256i final = _mm256_and_si256(cmp_i, val_i);

        int out[8];
        _mm256_storeu_si256((__m256i*)out, final);

        // --- for each candidate, atomically claim and record ---
        for (int b = 0; b < 8; ++b) {
          if (!out[b])
            continue;
          size_t nx = (size_t)nd.x[b], ny = (size_t)nd.y[b];
          if (burned_tracker.try_mark_burn(nx, ny)) {
            local_next.emplace_back(nx, ny);
            local_burned.emplace_back(nx, ny);
            ++new_burned;
          }
        }
      } // end omp for

      random_pool.refill();

#pragma omp critical
      {
        next_level.insert(next_level.end(), local_next.begin(), local_next.end());
        burned_tracker.burned_ids.insert(
            burned_tracker.burned_ids.end(), local_burned.begin(), local_burned.end()
        );
      }
    } // end omp parallel

    burned_tracker.burned_ids_steps.push_back(
        burned_tracker.burned_ids_steps.back() + new_burned
    );
    std::swap(current_level, next_level);
  }

  //
  // --- HERE: convert flat vector<char> → Matrix<bool> ---
  //
  Matrix<bool> result_map(n_col, n_row);
  for (size_t y = 0; y < n_row; ++y) {
    for (size_t x = 0; x < n_col; ++x) {
      size_t idx = y * n_col + x;
      result_map(x, y) = (burned_tracker.burned_bin[idx] != 0);
    }
  }

  return { n_col, n_row, std::move(result_map), std::move(burned_tracker.burned_ids),
           std::move(burned_tracker.burned_ids_steps) };
}