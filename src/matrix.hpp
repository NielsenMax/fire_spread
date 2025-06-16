#pragma once
#include <cstdint>

template <typename T> struct Matrix {
  size_t width;
  size_t height;
  std::vector<T> elems;

  Matrix(size_t width, size_t height) : width(width), height(height), elems(width * height) {
    // Ensure the vector's memory is aligned
    if (elems.size() > 0) {
      T* data = elems.data();
      if (reinterpret_cast<uintptr_t>(data) % 16 != 0) {
        // Reallocate with aligned memory if needed
        std::vector<T> aligned_elems(width * height);
        elems = std::move(aligned_elems);
      }
    }
  };

  const T operator[](std::pair<size_t, size_t> index) const {
    return elems[index.second * width + index.first];
  };

  T& operator[](std::pair<size_t, size_t> index) {
    return elems[index.second * width + index.first];
  };

  bool operator==(const Matrix& other) const {
    if (width != other.width || height != other.height) {
      return false;
    }

    for (size_t i = 0; i < width * height; i++) {
      if (elems[i] != other.elems[i]) {
        return false;
      }
    }

    return true;
  };
} __attribute__((aligned(16))); // Force 16-byte alignment for the Matrix struct

template <> struct Matrix<bool> {
  size_t width;
  size_t height;

  Matrix(size_t width, size_t height) : width(width), height(height), elems(width * height){};

  bool operator[](std::pair<size_t, size_t> index) const {
    return elems[index.second * width + index.first];
  };

  struct SmartReference {
    std::vector<bool>& values;
    size_t index;
    operator bool() const {
      return values[index];
    }
    SmartReference& operator=(bool const& other) {
      values[index] = other;
      return *this;
    }
  };

  SmartReference operator[](std::pair<size_t, size_t> index) {
    return SmartReference{ elems, index.second * width + index.first };
  }

  bool operator==(const Matrix& other) const {
    if (width != other.width || height != other.height) {
      return false;
    }

    for (size_t i = 0; i < width * height; i++) {
      if (elems[i] != other.elems[i]) {
        return false;
      }
    }

    return true;
  };

  std::vector<bool> elems;
};
