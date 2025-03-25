#!/bin/bash

# Check if binaries directory exists
if [ ! -d "./binaries" ]; then
    echo "Error: binaries directory not found. Run run_compilers.sh first."
    exit 1
fi

# Read data from scenarios.yaml
dataArr=($(yq '.data[]' scenarios.yaml))

# Run hyperfine for each data file
for data in "${dataArr[@]}"; do
    echo "Running benchmarks for data file: $data"
    
    # Get filename without path for CSV output
    data_filename=$(basename "$data")
    csv_output="results/benchmark_${data_filename%.*}.csv"
    
    # Build hyperfine command for this data file
    HYPERFINE_CMD="hyperfine --warmup 3 --export-csv $csv_output"
    
    # Add each binary in the binaries directory
    for binary in ./binaries/*; do
        # Get name without path and burned_probabilities_data_ prefix
        name=$(basename "$binary" | sed 's/burned_probabilities_data_//')
        # Use nice -n -20 to give highest priority to the process
        HYPERFINE_CMD+=" --command-name '$name' 'nice -n -20 $binary $data'"
    done

    # Execute hyperfine command for this data file
    eval "$HYPERFINE_CMD"
done
