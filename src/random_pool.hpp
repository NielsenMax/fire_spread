#pragma once

#include <vector>
#include <random>
#include <immintrin.h>  // For AVX2 intrinsics

class RandomPool {
private:
    std::vector<float> pool;
    size_t current_index;
    std::mt19937 generator;
    std::uniform_real_distribution<float> distribution;

public:
    RandomPool(size_t pool_size = 1024) 
        : pool(pool_size), current_index(0), 
          generator(std::random_device{}()),
          distribution(0.0f, 1.0f) {
        refill_pool();
    }

    void refill_pool() {
        for (size_t i = 0; i < pool.size(); i++) {
            pool[i] = distribution(generator);
        }
        current_index = 0;
    }

    float get_random() {
        if (current_index >= pool.size()) {
            refill_pool();
        }
        return pool[current_index++];
    }

    // Get 8 random numbers at once using AVX2
    __m256 get_random_avx() {
        if (current_index + 8 > pool.size()) {
            refill_pool();
        }
        __m256 result = _mm256_loadu_ps(&pool[current_index]);
        current_index += 8;
        return result;
    }
}; 