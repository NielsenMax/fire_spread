#!/bin/bash

compilers=(g++)
optimizations=(-O0 -O1 -O2 -O3)
extraOpts=(-march=native)
bin=./graphics/burned_probabilities_data
output=./graphics/results.csv
dataArr=(./data/1999_27j_S)

for compiler in "${compilers[@]}"; do
    for optimization in "${optimizations[@]}"; do
        for data in "${dataArr[@]}"; do
            make clean
            echo "Compiling with $compiler $optimization" >> ${output}
            make CXX=$compiler OPTFLAGS="$optimization"

            echo "Running $bin with data $data" >> ${output} 
            perf stat -x '|' -r 1 -e task-clock,cycles,instructions,branch-misses -o ${output} --append ./$bin $data
        done
    done
done
