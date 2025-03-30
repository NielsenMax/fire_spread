#!/bin/bash

# Find all metadata CSV files in the data directory
metadata_files=$(find ./data -name "*-metadata.csv")

# Create a temporary file to store results
temp_file=$(mktemp)

# Process each metadata file
for file in $metadata_files; do
    # Extract the first line (header) and second line (values)
    header=$(head -n 1 "$file" | tr -d '"')
    values=$(head -n 2 "$file" | tail -n 1 | tr -d '"')
    
    # Extract size_rows and size_cols using awk
    size_rows=$(echo "$values" | awk -F, '{print $1}')
    size_cols=$(echo "$values" | awk -F, '{print $2}')
    
    # Calculate the total size (rows * cols)
    total_size=$((size_rows * size_cols))
    
    # Store the result in the temporary file
    echo "$total_size $file" >> "$temp_file"
done

# Sort the results by total size (numeric sort)
echo "Metadata files sorted by size (rows * columns):"
sort -n "$temp_file" | awk '{print $2 " (size: " $1 ")"}'

# Clean up
rm "$temp_file"
