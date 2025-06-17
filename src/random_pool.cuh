// filepath: random_pool.cuh

#pragma once

#include <cstddef> // For size_t
#include <curand_kernel.h>

/**
 * @brief A resource-managing class for creating and handling a pool of
 * cuRAND states on the GPU.
 */
class RandomPoolCuda {
public:
  /**
     * @brief Constructs the pool and initializes random states on the GPU.
     * @param num_states The number of random generator states to create.
     * @param seed The seed for the random number generator.
     */
  RandomPoolCuda(size_t num_states, unsigned long long seed);

  /**
     * @brief Destructor that frees the allocated memory on the GPU.
     */
  ~RandomPoolCuda();

  // --- Move Semantics ---
  // We explicitly define how to "move" this object. This is required
  // for it to be stored correctly in a std::vector.

  /**
     * @brief Move constructor.
     */
  RandomPoolCuda(RandomPoolCuda&& other) noexcept;

  /**
     * @brief Move assignment operator.
     */
  RandomPoolCuda& operator=(RandomPoolCuda&& other) noexcept;

  // --- Deleted Copy Semantics ---
  // We explicitly forbid copying this object, because it manages a unique GPU
  // resource. Copying would be unsafe and lead to double-free errors.

  /**
     * @brief Deleted copy constructor.
     */
  RandomPoolCuda(const RandomPoolCuda&) = delete;

  /**
     * @brief Deleted copy assignment operator.
     */
  RandomPoolCuda& operator=(const RandomPoolCuda&) = delete;

  /**
     * @brief Gets the raw device pointer to the array of cuRAND states.
     * @return A device pointer to be passed to CUDA kernels.
     */
  curandState* get_states_ptr();

private:
  curandState* d_states; // Device pointer to the random states in Unified Memory
  size_t num_states;
};