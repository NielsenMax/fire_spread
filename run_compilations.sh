#!/bin/bash

# Read configuration from YAML
num_cases=$(yq '.cases | length' scenarios.yaml)
output=./graphics/results.csv
bin=./graphics/burned_probabilities_data

# Loop through each test case
for ((i=0; i<$num_cases; i++)); do
    compiler=$(yq ".cases[$i].compiler" scenarios.yaml)
    optimizations=($(yq ".cases[$i].optimizations[]" scenarios.yaml))
    extraOpts=($(yq ".cases[$i].default_opts[]" scenarios.yaml))
    dataArr=($(yq '.data[]' scenarios.yaml))

    # For each case, run all combinations
    for optimization in "${optimizations[@]}"; do
        for data in "${dataArr[@]}"; do
            make clean >/dev/null 2>&1
            echo "Compiling with $compiler $optimization ${extraOpts[*]}"
            make CXX=$compiler CXXFLAGS="${extraOpts[*]}" OPTFLAGS="$optimization" >/dev/null 2>&1

            # Run perf and capture output
            perf_output=$(perf stat -x '|' -r 1 -e task-clock,cycles,instructions,branch-misses ./$bin $data 2>&1 1>/dev/null)
            
            # Process each line of perf output and append compiler/optimization
            while IFS= read -r line; do
                if [[ $line =~ ^[0-9] ]]; then
                    echo "$line|$compiler|$optimization" >> ${output}
                fi
            done <<< "$perf_output"
        done
    done
done
