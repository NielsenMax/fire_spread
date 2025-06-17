// filepath: random_pool.cu

#include "random_pool.cuh"
#include <cstdio>  // For printf
#include <utility> // For std::swap in move assignment, though not strictly necessary

// Define the CUDA_CHECK macro here for error handling within this file
#define CUDA_CHECK(err)                                                                        \
  {                                                                                            \
    cudaError_t e = (err);                                                                     \
    if (e != cudaSuccess) {                                                                    \
      printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
      exit(EXIT_FAILURE);                                                                      \
    }                                                                                          \
  }

/**
 * @brief CUDA kernel to initialize an array of cuRAND states.
 * Each thread initializes one state.
 */
__global__ void
setup_rand_states_kernel(curandState* states, unsigned long long seed, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Initialize the state with a unique sequence based on thread index and seed
    curand_init(seed, idx, 0, &states[idx]);
  }
}

// --- Class Implementation ---

// Constructor
RandomPoolCuda::RandomPoolCuda(size_t num_states, unsigned long long seed)
    : d_states(nullptr), num_states(num_states) {
  if (num_states == 0)
    return;

  // Use cudaMallocManaged to allow for large allocations backed by system RAM
  CUDA_CHECK(cudaMallocManaged(&d_states, num_states * sizeof(curandState)));

  // Prefetch the memory to the GPU before launching the kernel that initializes it.
  int deviceId;
  CUDA_CHECK(cudaGetDevice(&deviceId));
  CUDA_CHECK(cudaMemPrefetchAsync(d_states, num_states * sizeof(curandState), deviceId, NULL));

  // Configure and launch the setup kernel
  const int block_size = 256;
  int num_blocks = (num_states + block_size - 1) / block_size;
  setup_rand_states_kernel<<<num_blocks, block_size>>>(d_states, seed, num_states);

  // Check for any errors during kernel launch
  CUDA_CHECK(cudaGetLastError());
}

// Destructor
RandomPoolCuda::~RandomPoolCuda() {
  // Free the memory if it was allocated.
  // cudaFree works for both cudaMalloc and cudaMallocManaged.
  if (d_states != nullptr) {
    cudaFree(d_states);
  }
}

// Move Constructor: Steals the resources from the 'other' object.
RandomPoolCuda::RandomPoolCuda(RandomPoolCuda&& other) noexcept
    : d_states(other.d_states), num_states(other.num_states) {
  // Leave the source object in a safe, empty state so its destructor won't double-free
  other.d_states = nullptr;
  other.num_states = 0;
}

// Move Assignment Operator: Handles moving resources from one object to another after construction.
RandomPoolCuda& RandomPoolCuda::operator=(RandomPoolCuda&& other) noexcept {
  // Protect against self-assignment (e.g., my_pool = std::move(my_pool))
  if (this != &other) {
    // Free our own existing resource before taking the new one
    if (d_states != nullptr) {
      cudaFree(d_states);
    }

    // Steal the resources from the 'other' object
    d_states = other.d_states;
    num_states = other.num_states;

    // Leave the source object in a safe, empty state
    other.d_states = nullptr;
    other.num_states = 0;
  }
  return *this;
}

// Getter for the device pointer
curandState* RandomPoolCuda::get_states_ptr() {
  return d_states;
}