#!/bin/bash

# Function to check for a parameter in the arguments
has_param() {
    local term="$1"
    shift
    for arg; do
        if [[ $arg == "$term" ]]; then
            return 0
        fi
    done
    return 1
}

# Variables
SRC_FILES="src/main.c src/network.c src/layer_dense.c src/utils.c libs/cJSON.c src/test_network.c"
INCLUDE_DIR="include/"
BUILD_DIR="build/"
OUTPUT_FILE="${BUILD_DIR}main"
CFLAGS="-O3 -g -march=native -funroll-loops -fopenmp -lm -ftree-vectorize -I ${INCLUDE_DIR} -I libs/"
PARALLEL_FLAG=""
DIAGNOSTIC_FLAG=""
SOCKET_FLAG=""

# Check for flags
if has_param "-parallel" "$@"; then
    echo "Compiling with OpenMP parallelization enabled..."
    PARALLEL_FLAG="-D ENABLE_PARALLEL -D NUM_THREADS=8"
fi

if has_param "-diag" "$@"; then
    echo "Compiling with debugging diagnostics flag enabled..."
    DIAGNOSTIC_FLAG="-fsanitize=address,undefined"
fi

# Run the Python script if the -python flag is set
if has_param "-python" "$@"; then
    echo "Running Python script: python/visualize_network.py"
    python python/visualize_network.py &
    if [[ $? -ne 0 ]]; then
        echo "Python script failed. Exiting."
        exit 1
    fi
    SOCKET_FLAG="-D ENABLE_SOCKET"
    sleep 2
fi

# Create build directory if it doesn't exist
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Default Compilation (Executable)
echo "Compiling the program..."
clang $CFLAGS $SOCKET_FLAG $PARALLEL_FLAG $DIAGNOSTIC_FLAG $SRC_FILES -o $OUTPUT_FILE

# Check if compilation was successful
if [[ $? -ne 0 ]]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

echo "Compilation successful. Output file: $OUTPUT_FILE"

# Run the program
echo "Running the program..."
time $OUTPUT_FILE 

echo "Program finished."
