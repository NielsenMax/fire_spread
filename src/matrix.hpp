#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <immintrin.h> // For AVX2 intrinsics
#include <vector>

// Aligned allocator for SIMD operations
template <typename T, size_t Alignment = 32> class AlignedAllocator {
public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U> struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  AlignedAllocator() = default;

  template <typename U, size_t OtherAlignment>
  AlignedAllocator(const AlignedAllocator<U, OtherAlignment>&) {}

  pointer allocate(size_type n) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<pointer>(ptr);
  }

  void deallocate(pointer p, size_type) {
    free(p);
  }

  bool operator==(const AlignedAllocator&) const {
    return true;
  }
  bool operator!=(const AlignedAllocator&) const {
    return false;
  }
};

template <typename T> struct Matrix {
  size_t width, height;
  std::vector<T, AlignedAllocator<T>> elems;

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

// Optimized specialization for bool matrix with vectorization support
template <> struct Matrix<bool> {
  size_t width, height;
  size_t bytes_per_row; // Cached value for faster row operations
  std::vector<std::byte, AlignedAllocator<std::byte>> elems; // Store 8 bools per byte

  Matrix(size_t width, size_t height)
      : width(width), height(height), bytes_per_row((width + 7) / 8),
        elems(bytes_per_row * height, std::byte{ 0 }) {}

  inline size_t index(size_t x, size_t y) const {
    return y * width + x;
  }

  inline size_t byte_index(size_t idx) const {
    return idx / 8;
  }

  inline uint8_t bit_index(size_t idx) const {
    return 1 << (idx % 8);
  }

  // Optimized access with prefetching for better cache utilization
  bool operator()(size_t x, size_t y) const {
    size_t idx = index(x, y);
    size_t byte_idx = byte_index(idx);

    // Prefetch the next few bytes to improve cache utilization
    if (byte_idx + 16 < elems.size()) {
      __builtin_prefetch(&elems[byte_idx + 16], 0, 0);
    }

    return static_cast<uint8_t>(elems[byte_idx]) & bit_index(idx);
  }

  struct SmartReference {
    std::vector<std::byte, AlignedAllocator<std::byte>>& values;
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

  // Vectorized operations for bulk access with improved memory access patterns

  // Set a row of boolean values using SIMD with prefetching
  void set_row(size_t y, bool value) {
    if (y >= height)
      return;

    const size_t row_byte_start = y * bytes_per_row;
    const std::byte fill_value =
        value ? static_cast<std::byte>(0xFF) : static_cast<std::byte>(0x00);

    // Process in chunks of 32 bytes for better cache utilization
    for (size_t i = 0; i < bytes_per_row; i += 32) {
      // Prefetch next chunk
      if (i + 64 < bytes_per_row) {
        __builtin_prefetch(&elems[row_byte_start + i + 32], 1, 0);
      }

      if (i + 32 <= bytes_per_row) {
        // Process 32 bytes at a time using AVX2
        __m256i fill_vec = _mm256_set1_epi8(static_cast<uint8_t>(fill_value));
        _mm256_store_si256(reinterpret_cast<__m256i*>(&elems[row_byte_start + i]), fill_vec);
      } else {
        // Process remaining bytes
        for (size_t j = i; j < bytes_per_row; j++) {
          elems[row_byte_start + j] = fill_value;
        }
      }
    }
  }

  // Check if a row contains any true values (vectorized with improved memory access)
  bool row_contains_true(size_t y) const {
    if (y >= height)
      return false;

    const size_t row_byte_start = y * bytes_per_row;

    // Process bytes in chunks of 32 for better cache utilization
    for (size_t i = 0; i < bytes_per_row; i += 32) {
      // Prefetch next chunk
      if (i + 64 < bytes_per_row) {
        __builtin_prefetch(&elems[row_byte_start + i + 32], 0, 0);
      }

      if (i + 32 <= bytes_per_row) {
        // Process 32 bytes at a time using AVX2
        __m256i row_data =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(&elems[row_byte_start + i]));
        if (_mm256_movemask_ps(_mm256_castsi256_ps(row_data)) != 0) {
          return true;
        }
      } else {
        // Process remaining bytes
        for (size_t j = i; j < bytes_per_row; j++) {
          if (static_cast<uint8_t>(elems[row_byte_start + j]) != 0) {
            return true;
          }
        }
      }
    }

    return false;
  }

  // Count the number of true values in a row (vectorized with improved memory access)
  size_t count_true_in_row(size_t y) const {
    if (y >= height)
      return 0;

    const size_t row_byte_start = y * bytes_per_row;
    size_t count = 0;

    // Process bytes in chunks of 32 for better cache utilization
    for (size_t i = 0; i < bytes_per_row; i += 32) {
      // Prefetch next chunk
      if (i + 64 < bytes_per_row) {
        __builtin_prefetch(&elems[row_byte_start + i + 32], 0, 0);
      }

      if (i + 32 <= bytes_per_row) {
        // Load 32 bytes
        __m256i row_data =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(&elems[row_byte_start + i]));

        // Extract bytes to a temporary array
        uint8_t temp_bytes[32] __attribute__((aligned(32)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(temp_bytes), row_data);

        // Count bits in each byte
        for (int j = 0; j < 32; j++) {
          count += __builtin_popcount(temp_bytes[j]);
        }
      } else {
        // Process remaining bytes
        for (size_t j = i; j < bytes_per_row; j++) {
          count += __builtin_popcount(static_cast<uint8_t>(elems[row_byte_start + j]));
        }
      }
    }

    return count;
  }

  // Reset all values to false (vectorized with improved memory access)
  void reset() {
    // Process in chunks of 32 bytes for better cache utilization
    for (size_t i = 0; i < elems.size(); i += 32) {
      // Prefetch next chunk
      if (i + 64 < elems.size()) {
        __builtin_prefetch(&elems[i + 32], 1, 0);
      }

      if (i + 32 <= elems.size()) {
        // Process 32 bytes at a time using AVX2
        __m256i zero_vec = _mm256_setzero_si256();
        _mm256_store_si256(reinterpret_cast<__m256i*>(&elems[i]), zero_vec);
      } else {
        // Process remaining bytes
        for (size_t j = i; j < elems.size(); j++) {
          elems[j] = static_cast<std::byte>(0);
        }
      }
    }
  }

  // Set a column of boolean values using SIMD with improved memory access
  void set_column(size_t x, bool value) {
    if (x >= width)
      return;

    const uint8_t bit_mask = bit_index(x);
    const uint8_t inv_bit_mask = ~bit_mask;
    const size_t byte_offset = byte_index(x);

    // Process rows in chunks for better cache utilization
    const size_t CHUNK_SIZE = 32;
    for (size_t chunk_start = 0; chunk_start < height; chunk_start += CHUNK_SIZE) {
      size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, height);

      // Prefetch next chunk
      if (chunk_end < height) {
        __builtin_prefetch(&elems[chunk_end * bytes_per_row + byte_offset], 1, 0);
      }

      for (size_t y = chunk_start; y < chunk_end; y++) {
        size_t byte_idx = y * bytes_per_row + byte_offset;
        if (value) {
          elems[byte_idx] |= static_cast<std::byte>(bit_mask);
        } else {
          elems[byte_idx] &= static_cast<std::byte>(inv_bit_mask);
        }
      }
    }
  }

  // Check if a column contains any true values (vectorized with improved memory access)
  bool column_contains_true(size_t x) const {
    if (x >= width)
      return false;

    const uint8_t bit_mask = bit_index(x);
    const size_t byte_offset = byte_index(x);

    // Process rows in chunks for better cache utilization
    const size_t CHUNK_SIZE = 32;
    for (size_t chunk_start = 0; chunk_start < height; chunk_start += CHUNK_SIZE) {
      size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, height);

      // Prefetch next chunk
      if (chunk_end < height) {
        __builtin_prefetch(&elems[chunk_end * bytes_per_row + byte_offset], 0, 0);
      }

      for (size_t y = chunk_start; y < chunk_end; y++) {
        size_t byte_idx = y * bytes_per_row + byte_offset;
        if (static_cast<uint8_t>(elems[byte_idx]) & bit_mask) {
          return true;
        }
      }
    }

    return false;
  }

  // Count the number of true values in a column (vectorized with improved memory access)
  size_t count_true_in_column(size_t x) const {
    if (x >= width)
      return 0;

    const uint8_t bit_mask = bit_index(x);
    const size_t byte_offset = byte_index(x);
    size_t count = 0;

    // Process rows in chunks for better cache utilization
    const size_t CHUNK_SIZE = 32;
    for (size_t chunk_start = 0; chunk_start < height; chunk_start += CHUNK_SIZE) {
      size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, height);

      // Prefetch next chunk
      if (chunk_end < height) {
        __builtin_prefetch(&elems[chunk_end * bytes_per_row + byte_offset], 0, 0);
      }

      for (size_t y = chunk_start; y < chunk_end; y++) {
        size_t byte_idx = y * bytes_per_row + byte_offset;
        if (static_cast<uint8_t>(elems[byte_idx]) & bit_mask) {
          count++;
        }
      }
    }

    return count;
  }

  // Bulk bit operations between matrices with improved memory access
  void bitwise_and(const Matrix<bool>& other) {
    if (width != other.width || height != other.height)
      return;

    // Process in chunks of 32 bytes for better cache utilization
    for (size_t i = 0; i < elems.size(); i += 32) {
      // Prefetch next chunk
      if (i + 64 < elems.size()) {
        __builtin_prefetch(&elems[i + 32], 1, 0);
        __builtin_prefetch(&other.elems[i + 32], 0, 0);
      }

      if (i + 32 <= elems.size()) {
        // Process 32 bytes at a time using AVX2
        __m256i this_data = _mm256_load_si256(reinterpret_cast<const __m256i*>(&elems[i]));
        __m256i other_data =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(&other.elems[i]));
        __m256i result = _mm256_and_si256(this_data, other_data);
        _mm256_store_si256(reinterpret_cast<__m256i*>(&elems[i]), result);
      } else {
        // Process remaining bytes
        for (size_t j = i; j < elems.size(); j++) {
          elems[j] &= other.elems[j];
        }
      }
    }
  }

  void bitwise_or(const Matrix<bool>& other) {
    if (width != other.width || height != other.height)
      return;

    // Process in chunks of 32 bytes for better cache utilization
    for (size_t i = 0; i < elems.size(); i += 32) {
      // Prefetch next chunk
      if (i + 64 < elems.size()) {
        __builtin_prefetch(&elems[i + 32], 1, 0);
        __builtin_prefetch(&other.elems[i + 32], 0, 0);
      }

      if (i + 32 <= elems.size()) {
        // Process 32 bytes at a time using AVX2
        __m256i this_data = _mm256_load_si256(reinterpret_cast<const __m256i*>(&elems[i]));
        __m256i other_data =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(&other.elems[i]));
        __m256i result = _mm256_or_si256(this_data, other_data);
        _mm256_store_si256(reinterpret_cast<__m256i*>(&elems[i]), result);
      } else {
        // Process remaining bytes
        for (size_t j = i; j < elems.size(); j++) {
          elems[j] |= other.elems[j];
        }
      }
    }
  }

  void bitwise_xor(const Matrix<bool>& other) {
    if (width != other.width || height != other.height)
      return;

    // Process in chunks of 32 bytes for better cache utilization
    for (size_t i = 0; i < elems.size(); i += 32) {
      // Prefetch next chunk
      if (i + 64 < elems.size()) {
        __builtin_prefetch(&elems[i + 32], 1, 0);
        __builtin_prefetch(&other.elems[i + 32], 0, 0);
      }

      if (i + 32 <= elems.size()) {
        // Process 32 bytes at a time using AVX2
        __m256i this_data = _mm256_load_si256(reinterpret_cast<const __m256i*>(&elems[i]));
        __m256i other_data =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(&other.elems[i]));
        __m256i result = _mm256_xor_si256(this_data, other_data);
        _mm256_store_si256(reinterpret_cast<__m256i*>(&elems[i]), result);
      } else {
        // Process remaining bytes
        for (size_t j = i; j < elems.size(); j++) {
          elems[j] ^= other.elems[j];
        }
      }
    }
  }

  // Count total number of true values in the matrix (vectorized with improved memory access)
  size_t count_true() const {
    size_t count = 0;
    const size_t total_bytes = bytes_per_row * height;

    // Process bytes in chunks of 32 for better cache utilization
    for (size_t i = 0; i < total_bytes; i += 32) {
      // Prefetch next chunk
      if (i + 64 < total_bytes) {
        __builtin_prefetch(&elems[i + 32], 0, 0);
      }

      if (i + 32 <= total_bytes) {
        // Load 32 bytes
        __m256i data = _mm256_load_si256(reinterpret_cast<const __m256i*>(&elems[i]));

        // Extract bytes to a temporary array
        uint8_t temp_bytes[32] __attribute__((aligned(32)));
        _mm256_store_si256(reinterpret_cast<__m256i*>(temp_bytes), data);

        // Count bits in each byte
        for (int j = 0; j < 32; j++) {
          count += __builtin_popcount(temp_bytes[j]);
        }
      } else {
        // Process remaining bytes
        for (size_t j = i; j < total_bytes; j++) {
          count += __builtin_popcount(static_cast<uint8_t>(elems[j]));
        }
      }
    }

    return count;
  }

  // Set a region of the matrix to a specific value (vectorized with improved memory access)
  void set_region(size_t start_x, size_t start_y, size_t end_x, size_t end_y, bool value) {
    if (start_x >= width || start_y >= height || end_x > width || end_y > height ||
        start_x >= end_x || start_y >= end_y)
      return;

    const uint8_t bit_mask = value ? 0xFF : 0x00;

    // Process rows in chunks for better cache utilization
    const size_t CHUNK_SIZE = 32;
    for (size_t chunk_start = start_y; chunk_start < end_y; chunk_start += CHUNK_SIZE) {
      size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, end_y);

      // Prefetch next chunk
      if (chunk_end < end_y) {
        __builtin_prefetch(&elems[chunk_end * bytes_per_row], 1, 0);
      }

      for (size_t y = chunk_start; y < chunk_end; y++) {
        size_t row_start = y * width;
        size_t row_end = row_start + end_x;

        // Handle partial bytes at the start of the row
        size_t first_byte = byte_index(row_start + start_x);
        size_t last_byte = byte_index(row_end - 1);

        if (first_byte == last_byte) {
          // All bits are in the same byte
          uint8_t mask = 0;
          for (size_t x = start_x; x < end_x; x++) {
            mask |= bit_index(index(x, y));
          }

          if (value) {
            elems[first_byte] |= static_cast<std::byte>(mask);
          } else {
            elems[first_byte] &= static_cast<std::byte>(~mask);
          }
        } else {
          // Handle first byte
          uint8_t first_mask = 0;
          for (size_t x = start_x; x < end_x && byte_index(index(x, y)) == first_byte; x++) {
            first_mask |= bit_index(index(x, y));
          }

          if (value) {
            elems[first_byte] |= static_cast<std::byte>(first_mask);
          } else {
            elems[first_byte] &= static_cast<std::byte>(~first_mask);
          }

          // Handle middle bytes (full bytes)
          for (size_t byte = first_byte + 1; byte < last_byte; byte++) {
            elems[byte] = static_cast<std::byte>(bit_mask);
          }

          // Handle last byte
          uint8_t last_mask = 0;
          for (size_t x = end_x - 1; x >= start_x && byte_index(index(x, y)) == last_byte;
               x--) {
            last_mask |= bit_index(index(x, y));
          }

          if (value) {
            elems[last_byte] |= static_cast<std::byte>(last_mask);
          } else {
            elems[last_byte] &= static_cast<std::byte>(~last_mask);
          }
        }
      }
    }
  }

  // Check if any value in a region is true (vectorized with improved memory access)
  bool region_contains_true(size_t start_x, size_t start_y, size_t end_x, size_t end_y) const {
    if (start_x >= width || start_y >= height || end_x > width || end_y > height ||
        start_x >= end_x || start_y >= end_y)
      return false;

    // Process rows in chunks for better cache utilization
    const size_t CHUNK_SIZE = 32;
    for (size_t chunk_start = start_y; chunk_start < end_y; chunk_start += CHUNK_SIZE) {
      size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, end_y);

      // Prefetch next chunk
      if (chunk_end < end_y) {
        __builtin_prefetch(&elems[chunk_end * bytes_per_row], 0, 0);
      }

      for (size_t y = chunk_start; y < chunk_end; y++) {
        for (size_t x = start_x; x < end_x; x++) {
          if (operator()(x, y)) {
            return true;
          }
        }
      }
    }

    return false;
  }

  // Count true values in a region (vectorized with improved memory access)
  size_t
  count_true_in_region(size_t start_x, size_t start_y, size_t end_x, size_t end_y) const {
    if (start_x >= width || start_y >= height || end_x > width || end_y > height ||
        start_x >= end_x || start_y >= end_y)
      return 0;

    size_t count = 0;

    // Process rows in chunks for better cache utilization
    const size_t CHUNK_SIZE = 32;
    for (size_t chunk_start = start_y; chunk_start < end_y; chunk_start += CHUNK_SIZE) {
      size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, end_y);

      // Prefetch next chunk
      if (chunk_end < end_y) {
        __builtin_prefetch(&elems[chunk_end * bytes_per_row], 0, 0);
      }

      for (size_t y = chunk_start; y < chunk_end; y++) {
        for (size_t x = start_x; x < end_x; x++) {
          if (operator()(x, y)) {
            count++;
          }
        }
      }
    }

    return count;
  }
};