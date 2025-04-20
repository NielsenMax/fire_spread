// landscape.cpp (Fixed std::stoul call)
#include "landscape.hpp"

#include <cstddef>
#include <cstdlib> // For atoi, atof
#include <fstream>
#include <stdexcept> // For runtime_error
#include <string>
#include <vector> // Included via landscape.hpp but good practice

// Constructor for empty landscape
Landscape::Landscape(size_t width, size_t height)
    : width(width), height(height),
      // Initialize all matrices
      burnable_m(width, height), // Matrix<bool> default constructor likely sets to false
      elevations(width, height), wind_directions(width, height),
      vegetation_types(width, height), // Defaults underlying uint8_t to 0 (MATORRAL)
      fwis(width, height), aspects(width, height) {}

// Constructor loading from files
Landscape::Landscape(const std::string& metadata_filename, const std::string& data_filename)
    : width(0), height(0), // Initialize dimensions
      // Initialize matrices with temporary size 0, will resize after reading metadata
      burnable_m(0, 0), elevations(0, 0), wind_directions(0, 0), vegetation_types(0, 0),
      fwis(0, 0), aspects(0, 0) {
  std::ifstream metadata_file(metadata_filename);
  if (!metadata_file.is_open()) {
    throw std::runtime_error("Can't open metadata file: " + metadata_filename);
  }

  CSVIterator metadata_csv(metadata_file);
  // Check if iterator is valid before incrementing (handles empty file)
  if (metadata_csv != CSVIterator()) {
    ++metadata_csv; // Skip header
  }

  if (metadata_csv == CSVIterator() || (*metadata_csv).size() < 2) {
    metadata_file.close(); // Close file before throwing
    throw std::runtime_error(
        "Invalid metadata file content or header missing: " + metadata_filename
    );
  }

  // Read dimensions using stoul for size_t, converting argument to std::string
  try {
    // *** FIXED: Explicitly construct std::string ***
    width = std::stoul(std::string((*metadata_csv)[0]));
    height = std::stoul(std::string((*metadata_csv)[1]));
    // *** END FIX ***
  } catch (const std::exception& e) {
    metadata_file.close();
    throw std::runtime_error(
        "Error parsing dimensions in metadata file '" + metadata_filename + "': " + e.what()
    );
  }
  metadata_file.close();

  if (width == 0 || height == 0) {
    throw std::runtime_error(
        "Invalid dimensions (width=" + std::to_string(width) +
        ", height=" + std::to_string(height) + ") read from metadata: " + metadata_filename
    );
  }

  // --- Resize matrices to correct dimensions ---
  burnable_m = Matrix<bool>(width, height);
  elevations = Matrix<float>(width, height);
  wind_directions = Matrix<float>(width, height);
  vegetation_types = Matrix<uint8_t>(width, height);
  fwis = Matrix<float>(width, height);
  aspects = Matrix<float>(width, height);
  // --- End Resizing ---

  std::ifstream landscape_file(data_filename);
  if (!landscape_file.is_open()) {
    throw std::runtime_error("Can't open landscape file: " + data_filename);
  }

  CSVIterator loop_csv(landscape_file);
  // Check if file is not empty before skipping header
  if (loop_csv != CSVIterator()) {
    ++loop_csv; // Skip header row
  }

  size_t row_num = 1; // Start after header
  for (size_t j = 0; j < height; j++) {
    for (size_t i = 0; i < width; i++, row_num++) {
      if (loop_csv == CSVIterator()) {
        landscape_file.close();
        throw std::runtime_error(
            "Insufficient rows in landscape file '" + data_filename + "'. Expected " +
            std::to_string(width * height) + ", stopped at row " + std::to_string(row_num)
        );
      }
      const auto& row = *loop_csv;
      if (row.size() < 8) {
        landscape_file.close();
        throw std::runtime_error(
            "Insufficient columns in landscape file '" + data_filename + "' at file row " +
            std::to_string(row_num + 1) + " (data for cell " + std::to_string(i) + "," +
            std::to_string(j) + ")"
        );
      }

      // Use try-catch for parsing robustness
      try {
        int veg_subalpine = std::atoi(row[0].data());
        int veg_wet = std::atoi(row[1].data());
        int veg_dry = std::atoi(row[2].data());
        float cell_fwi = std::atof(row[3].data());
        float cell_aspect = std::atof(row[4].data());
        float cell_wind_dir = std::atof(row[5].data());
        float cell_elev = std::atof(row[6].data());
        bool cell_burnable = (std::atoi(row[7].data()) != 0);

        VegetationType veg_type = MATORRAL; // Default
        if (veg_subalpine == 1) {
          veg_type = SUBALPINE;
        } else if (veg_wet == 1) {
          veg_type = WET;
        } else if (veg_dry == 1) {
          veg_type = DRY;
        }

        // Assign values to SoA matrices
        elevations(i, j) = cell_elev;
        wind_directions(i, j) = cell_wind_dir;
        vegetation_types(i, j) = static_cast<uint8_t>(veg_type);
        fwis(i, j) = cell_fwi;
        aspects(i, j) = cell_aspect;
        burnable_m(i, j) = cell_burnable;

      } catch (const std::exception& e) {
        landscape_file.close();
        throw std::runtime_error(
            "Error parsing data in landscape file '" + data_filename + "' at file row " +
            std::to_string(row_num + 1) + ": " + e.what()
        );
      }

      ++loop_csv; // Move to next row in CSV
    }
  }

  if (loop_csv != CSVIterator()) {
    // Optional: Warn or error if there are more rows than expected
  }

  landscape_file.close();
}