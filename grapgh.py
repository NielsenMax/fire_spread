import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For color mapping
import numpy as np
import os # To check file existence

# --- Configuration ---
# <<< IMPORTANT: Change this to the actual path of your CSV file >>>
CSV_FILE_PATH = 'joined.csv'

# --- Validate File Path ---
if not os.path.exists(CSV_FILE_PATH):
    print(f"Error: File not found at '{CSV_FILE_PATH}'")
    print("Please update the CSV_FILE_PATH variable in the script.")
    # Create a dummy file for demonstration if it doesn't exist
    print("Creating a dummy 'your_benchmark_data.csv' for demonstration purposes.")
    dummy_data = {
        'benchmark': ['Bench1', 'Bench1', 'Bench2', 'Bench2', 'Bench2', 'Bench3', 'Bench3'],
        'command': ['cmdA', 'cmdB', 'cmdA', 'cmdB', 'cmdC', 'cmdA', 'cmdB'],
        'area': [100, 100, 300, 300, 300, 150, 150], # Modified areas for sorting demo
        'times faster': [1.0, 1.5, 1.2, 2.1, 1.8, 0.9, 1.3]
    }
    try:
        pd.DataFrame(dummy_data).to_csv(CSV_FILE_PATH, index=False)
        print(f"Dummy file created at '{CSV_FILE_PATH}'. Please replace it with your actual data.")
    except Exception as e:
        print(f"Could not create dummy file: {e}")
        exit() # Exit if dummy file creation fails

# --- Load Data ---
try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(CSV_FILE_PATH)

    # --- Data Validation and Preparation ---
    # Check if required columns exist (added 'area')
    required_columns = ['benchmark', 'command', 'times faster', 'area']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"Error: Missing required columns in CSV: {missing}")
        exit()

    # Ensure 'times faster' and 'area' are numeric, convert errors to NaN
    df['times faster'] = pd.to_numeric(df['times faster'], errors='coerce')
    df['area'] = pd.to_numeric(df['area'], errors='coerce')

    # Remove rows where 'times faster' or 'area' could not be converted or are missing
    original_rows = len(df)
    df.dropna(subset=['times faster', 'area'], inplace=True)
    if len(df) < original_rows:
        print(f"Warning: Removed {original_rows - len(df)} rows due to missing or non-numeric 'times faster' or 'area' values.")

    # Check if DataFrame is empty after cleaning
    if df.empty:
        print("Error: No valid data remaining after cleaning. Please check your CSV file.")
        exit()

    # --- Sort Benchmarks by Area (New Step) ---
    # Group by benchmark and get the first area value (assuming area is consistent per benchmark)
    # Use mean() instead of first() if area might vary slightly per command but you want average area for sorting
    benchmark_area = df.groupby('benchmark')['area'].first().reset_index()
    # Sort the benchmarks based on area (ascending)
    benchmark_area.sort_values('area', ascending=True, inplace=True)
    # Get the sorted list of benchmark names
    benchmarks_sorted_by_area = benchmark_area['benchmark'].tolist()

    # Use this area-sorted list for plotting order
    benchmarks = benchmarks_sorted_by_area

    # Get unique commands for coloring and legend (sort for consistency)
    commands = df['command'].unique()
    commands.sort()

except FileNotFoundError:
    print(f"Error: File not found at '{CSV_FILE_PATH}'")
    exit()
except Exception as e:
    print(f"An error occurred during data loading or preparation: {e}")
    exit()

# --- Plotting ---
fig, ax = plt.subplots(figsize=(max(10, len(benchmarks) * 1.2), 7)) # Dynamic figure width

# Assign unique colors to each command using a colormap
colors = cm.get_cmap('tab10', len(commands))
color_map = {command: colors(i) for i, command in enumerate(commands)}

# Calculate the positions for each benchmark on the x-axis
x_positions = np.arange(len(benchmarks))
bar_width = 0.6 # Adjust bar width as needed

# Store handles for the legend
legend_handles = {}

# Iterate through each unique benchmark (now sorted by area)
for i, benchmark in enumerate(benchmarks):
    # Filter data for the current benchmark
    bench_data = df[df['benchmark'] == benchmark].copy()

    # --- Crucial Step for Superposition (Modified) ---
    # Sort the commands within this benchmark by 'times faster' in DESCENDING order
    # This ensures taller bars are plotted first and shorter bars are plotted over them
    bench_data.sort_values('times faster', ascending=False, inplace=True) # Changed to False

    # Plot a bar for each command within this benchmark at the SAME x-position
    for _, row in bench_data.iterrows():
        command = row['command']
        times_faster_value = row['times faster']
        color = color_map[command]

        # Plot the bar
        bar = ax.bar(x_positions[i],  # X-position for the benchmark
                     times_faster_value, # Height of the bar
                     width=bar_width,
                     color=color,
                     label=command) # Assign label for legend

        # Store the handle for the legend (only need one per command)
        if command not in legend_handles:
            legend_handles[command] = bar

# --- Plot Customization ---
ax.set_xlabel('Benchmark (Ordenado por Area)', fontsize=12) # Updated label
ax.set_ylabel('Cantidad de veces mas rapido', fontsize=12)
ax.set_title('Rendimiento de Benchmark: Cantidad de veces mas rapido por comando (Superpuesto)', fontsize=14)

# Set x-axis ticks and labels using the area-sorted benchmark list
ax.set_xticks(x_positions)
ax.set_xticklabels(benchmarks, rotation=45, ha='right')

# Set y-axis limits (optional, start from 0)
ax.set_ylim(bottom=0)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Create the legend using the stored handles to avoid duplicates
# Place legend outside the plot area
ax.legend(legend_handles.values(), legend_handles.keys(), title='Commands', bbox_to_anchor=(1.02, 1), loc='upper left')

# Adjust layout to prevent labels overlapping and make space for legend
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust right boundary to fit legend
plt.show()

print("-" * 30)
print("Graph generation complete.")
print(f"Processed {len(df)} valid data rows.")
print(f"Found {len(benchmarks)} unique benchmarks (sorted by area).")
print(f"Found {len(commands)} unique commands.")
print("-" * 30)

