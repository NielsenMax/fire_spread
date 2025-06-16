#pragma once

#include "csv.hpp"
#include "matrix.hpp"

// enum of vegetation type between: matorral, subalpine, wet, dry
enum VegetationType {
  MATORRAL,
  SUBALPINE,
  WET,
  DRY
} __attribute__((packed)); // packed so that it takes only 1 byte

static_assert( sizeof(VegetationType) == 1 );

struct Cell {
  float elevation;
  float wind_direction;
  float fwi;
  float aspect;
  bool burnable;
  VegetationType vegetation_type;
  char padding[2]; // Add padding to align to 16 bytes
} __attribute__((aligned(16))); // Force 16-byte alignment for vectorization

struct Landscape {
  size_t width;
  size_t height;

  Landscape(size_t width, size_t height);
  Landscape(std::string metadata_filename, std::string data_filename);

  Cell operator[](std::pair<size_t, size_t> index) const;
  Cell& operator[](std::pair<size_t, size_t> index);

  Matrix<Cell> cells;
};
