#pragma once

#include <string>
#include <utility> // For std::pair
#include <vector>

#include "csv.hpp"    // Assumed to be included for CSVIterator used in .cpp
#include "matrix.hpp" // Use the provided Matrix template

// Enum of vegetation type
// Explicitly use uint8_t for underlying type for defined size and Matrix compatibility
enum VegetationType : uint8_t { MATORRAL = 0, SUBALPINE = 1, WET = 2, DRY = 3 };

// Cell struct can remain for conceptual representation or specific functions
struct Cell {
  float elevation;
  float wind_direction;
  bool burnable;
  VegetationType vegetation_type;
  float fwi;
  float aspect;
};

// Landscape class using Struct-of-Arrays (SoA)
struct Landscape {
  size_t width;
  size_t height;

  // --- SoA Data Storage ---
  Matrix<bool> burnable_m; // Using optimized Matrix<bool>
  Matrix<float> elevations;
  Matrix<float> wind_directions;
  Matrix<uint8_t> vegetation_types; // Store enum's underlying type
  Matrix<float> fwis;
  Matrix<float> aspects;
  // --- End SoA Data Storage ---

  // Constructor to initialize empty landscape
  Landscape(size_t width, size_t height);

  // Constructor loading from files (use const&)
  Landscape(const std::string& metadata_filename, const std::string& data_filename);

  // --- Direct Accessors for SoA data ---
  // Const versions
  inline float get_elevation(size_t x, size_t y) const {
    return elevations(x, y);
  }
  inline float get_wind_direction(size_t x, size_t y) const {
    return wind_directions(x, y);
  }
  inline bool is_burnable(size_t x, size_t y) const {
    return burnable_m(x, y);
  }
  inline VegetationType get_vegetation_type(size_t x, size_t y) const {
    return static_cast<VegetationType>(vegetation_types(x, y));
  }
  inline float get_fwi(size_t x, size_t y) const {
    return fwis(x, y);
  }
  inline float get_aspect(size_t x, size_t y) const {
    return aspects(x, y);
  }

  // Method to get all data for one cell (requires multiple lookups)
  Cell get_cell_data(size_t x, size_t y) const {
    return { get_elevation(x, y), get_wind_direction(x, y),
             is_burnable(x, y),   get_vegetation_type(x, y),
             get_fwi(x, y),       get_aspect(x, y) };
  }

  // Removed the old operator[] that returned Cell or Cell&
};