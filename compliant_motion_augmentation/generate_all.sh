#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the list of modes to iterate over
MODES=("forcefield" "collision-emulator" "zero-wrench")

NUM_MODE_FILES=(40 40 5)

# The base directory for all output
BASE_OUTPUT_DIR="release_examples"

# Loop through each mode
for i in "${!MODES[@]}"; do
  mode=${MODES[$i]}
  num_files=${NUM_MODE_FILES[$i]}
  echo "=================================================="
  echo "  Generating ${num_files} files for mode: $mode"
  echo "=================================================="

  # Create the output directory for the current mode
  mkdir -p "${BASE_OUTPUT_DIR}/${mode}"

  # Motion 1: stand
  python mink_generator_ff.py generate-data \
    --motion_path ../datasets/motions_csv/stand.csv \
    --force_mode "$mode" \
    --num_files $num_files \
    --output_dir "${BASE_OUTPUT_DIR}/${mode}/stand"

  # Motion 2: tpose
  python mink_generator_ff.py generate-data \
    --motion_path ../datasets/motions_csv/tpose.csv \
    --force_mode "$mode" \
    --num_files $num_files \
    --output_dir "${BASE_OUTPUT_DIR}/${mode}/tpose"

  # Motion 3: boxpick
  python mink_generator_ff.py generate-data \
    --motion_path ../datasets/motions_csv/boxpick.csv \
    --force_mode "$mode" \
    --num_files $num_files \
    --output_dir "${BASE_OUTPUT_DIR}/${mode}/boxpick"

  # Motion 4: walk
  python mink_generator_ff.py generate-data \
    --motion_path ../datasets/motions_csv/walk.csv \
    --force_mode "$mode" \
    --num_files $num_files \
    --output_dir "${BASE_OUTPUT_DIR}/${mode}/walk"

  # Motion 5: pour
  python mink_generator_ff.py generate-data \
    --motion_path ../datasets/motions_csv/pour.csv \
    --force_mode "$mode" \
    --num_files $num_files \
    --output_dir "${BASE_OUTPUT_DIR}/${mode}/pour"

done

echo "=================================================="
echo "  All commands completed for all modes."
echo "=================================================="
