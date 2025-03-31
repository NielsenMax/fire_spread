#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

template <typename T> struct Matrix {
  size_t width, height;
  std::vector<T> elems;

  Matrix(size_t width, size_t height)
      : width(width), height(height), elems(width * height, T{}) {}

  inline size_t index(size_t x, size_t y) const {
    return y * width + x;
  }

  const T& operator()(size_t x, size_t y) const {
    return elems[index(x, y)];
  }

  T& operator()(size_t x, size_t y) {
    return elems[index(x, y)];
  }

  bool operator==(const Matrix& other) const {
    return width == other.width && height == other.height &&
           std::equal(elems.begin(), elems.end(), other.elems.begin());
  }
};

// Optimized specialization for bool matrix
template <> struct Matrix<bool> {
  size_t width, height;
  std::vector<std::byte> elems; // Store 8 bools per byte

  Matrix(size_t width, size_t height)
      : width(width), height(height), elems((width * height + 7) / 8, std::byte{ 0 }) {}

  inline size_t index(size_t x, size_t y) const {
    return y * width + x;
  }

  inline size_t byte_index(size_t idx) const {
    return idx / 8;
  }

  inline uint8_t bit_index(size_t idx) const {
    return 1 << (idx % 8);
  }

  bool operator()(size_t x, size_t y) const {
    size_t idx = index(x, y);
    return static_cast<uint8_t>(elems[byte_index(idx)]) & bit_index(idx);
  }

  struct SmartReference {
    std::vector<std::byte>& values;
    size_t idx;

    operator bool() const {
      return static_cast<uint8_t>(values[idx / 8]) & (1 << (idx % 8));
    }

    SmartReference& operator=(bool val) {
      if (val)
        values[idx / 8] |= static_cast<std::byte>(1 << (idx % 8));
      else
        values[idx / 8] &= static_cast<std::byte>(~(1 << (idx % 8)));
      return *this;
    }
  };

  SmartReference operator()(size_t x, size_t y) {
    return { elems, index(x, y) };
  }

  bool operator==(const Matrix& other) const {
    return width == other.width && height == other.height && elems == other.elems;
  }
};