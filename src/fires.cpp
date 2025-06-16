#include "fires.hpp"

#include <fstream>
#include <omp.h> // Include OpenMP header

#include "csv.hpp" // Assuming this is used by read_fire, not get_fire_stats directly
#include "landscape.hpp"
#include "matrix.hpp"

Fire read_fire(size_t width, size_t height, std::string filename) {

  std::ifstream burned_ids_file(filename);

  if (!burned_ids_file.is_open()) {
    throw std::runtime_error("Can't open landscape file");
  }

  CSVIterator loop(burned_ids_file);
  loop++;

  Matrix<bool> burned_layer(width, height);

  std::vector<std::pair<size_t, size_t>> burned_ids;

  for (; loop != CSVIterator(); ++loop) {
    if (loop->size() < 2) {
      throw std::runtime_error("Invalid fire file");
    }
    size_t x = atoi((*loop)[0].data());
    size_t y = atoi((*loop)[1].data());
    if (x >= width || y >= height) {
      throw std::runtime_error("Invalid fire file");
    }
    burned_layer[{ x, y }] = true;
    burned_ids.push_back({ x, y });
  }

  burned_ids_file.close();

  return { width, height, burned_layer, burned_ids, {} };
}

FireStats get_fire_stats(const Fire& fire, const Landscape& landscape) {
  FireStats stats = { 0, 0, 0, 0 };

  // Use OpenMP reduction for summing up stats
  // Declare local counters for reduction to avoid false sharing if stats struct is small
  size_t local_matorral = 0;
  size_t local_subalpine = 0;
  size_t local_wet = 0;
  size_t local_dry = 0;

  // The loop iterates over burned_ids, which can be done in parallel.
  // landscape is read-only.
  #pragma omp parallel for reduction(+:local_matorral, local_subalpine, local_wet, local_dry) schedule(static)
  for (size_t i = 0; i < fire.burned_ids.size(); ++i) {
    const auto& coord = fire.burned_ids[i];
    // Accessing landscape is const, so it's thread-safe for reading.
    Cell cell = landscape[coord]; // Assuming operator[] is thread-safe for const access

    if (cell.vegetation_type == SUBALPINE) {
      local_subalpine++;
    } else if (cell.vegetation_type == WET) {
      local_wet++;
    } else if (cell.vegetation_type == DRY) {
      local_dry++;
    } else { // MATORRAL
      local_matorral++;
    }
  }

  stats.counts_veg_matorral = local_matorral;
  stats.counts_veg_subalpine = local_subalpine;
  stats.counts_veg_wet = local_wet;
  stats.counts_veg_dry = local_dry;

  return stats;
}
