#!/bin/bash

# Read configuration from YAML
num_cases=$(yq '.cases | length' scenarios.yaml)
bin=burned_probabilities_data
binaries_dir="./binaries"

# Create binaries directory if it doesn't exist
mkdir -p "$binaries_dir"

# Loop through each test case
# First compile all binaries
declare -a commands=()
for ((i=0; i<$num_cases; i++)); do
    compiler=$(yq ".cases[$i].compiler" scenarios.yaml)
    optimizations=($(yq ".cases[$i].optimizations[]" scenarios.yaml))
    extraOpts=($(yq ".cases[$i].default_opts[]" scenarios.yaml))
    dataArr=($(yq '.data[]' scenarios.yaml))

    for optimization in "${optimizations[@]}"; do
        for data in "${dataArr[@]}"; do
            # Create unique binary name for this combination
            binary_name="${bin}_${compiler}_${optimization// /_}"
            make clean >/dev/null 2>&1
            echo "Compiling with $compiler $optimization ${extraOpts[*]}"
            make specific CXX=$compiler CXXFLAGS="${extraOpts[*]}" OPTFLAGS="$optimization" BINARY_NAME="$binary_name"
            
            # Move binary to binaries directory
            # mv "./graphics/$binary_name" "$binaries_dir/"
            if [ ! -f "$binaries_dir/$binary_name" ]; then
                echo "Error: Binary $binary_name was not created"
                exit 1
            fi
        done
    done
done

echo "All binaries compiled. Run run_benchmarks.sh to execute benchmarks."
