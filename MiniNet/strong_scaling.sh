#!/bin/bash

# Variables
SRC_FILES="src/main.c src/network.c src/layer_dense.c src/utils.c libs/cJSON.c src/test_network.c"
INCLUDE_DIR="include/"
BUILD_DIR="build/"
OUTPUT_FILE="${BUILD_DIR}main"
CFLAGS="-O3 -march=native -funroll-loops -D ENABLE_PARALLEL -fopenmp -lm -ftree-vectorize -I ${INCLUDE_DIR} -I libs/"

# Thread counts to test
THREAD_COUNTS=(1 2 4 8)

# Create build directory if it doesn't exist
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Loop through each thread count
for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
    echo "Compiling with NUM_THREADS=${NUM_THREADS}..."

    # Compile with NUM_THREADS defined at compile time
    clang $CFLAGS -D STUDY_TIMING -D NUM_THREADS=$NUM_THREADS $SRC_FILES -o $OUTPUT_FILE
    if [[ $? -ne 0 ]]; then
        echo "Compilation failed for NUM_THREADS=${NUM_THREADS}. Exiting."
        exit 1
    fi

    echo "Compilation successful for NUM_THREADS=${NUM_THREADS}"
    echo "Running..."
    $OUTPUT_FILE
done
