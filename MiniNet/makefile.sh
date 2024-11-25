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
SRC_FILES="src/main.c src/network.c src/layer_dense.c src/utils.c"
INCLUDE_DIR="include/"
BUILD_DIR="build/"
OUTPUT_FILE="${BUILD_DIR}main"
SHARED_LIB_OUTPUT="${BUILD_DIR}libnn.so"  # Shared library output file
CFLAGS="-O3 -march=native -funroll-loops -lm -ftree-vectorize -I ${INCLUDE_DIR}"
PARALLEL_FLAG=""
DIAGNOSTIC_FLAG=""

# Check for flags
if has_param "-parallel" "$@"; then
    echo "Compiling with OpenMP parallelization enabled..."
    PARALLEL_FLAG="-fopenmp -D ENABLE_PARALLEL"
fi

if has_param "-diag" "$@"; then
    echo "Compiling with sanitizer diagnostics enabled..."
    DIAGNOSTIC_FLAG="-fsanitize=address -g"
fi

if has_param "-shared" "$@"; then
    echo "Compiling into shared library..."

    # Modify the compile command to create a shared library instead of an executable
    OUTPUT_FILE="$SHARED_LIB_OUTPUT"  # Change the output to the shared library

    # Compile the shared library (using gcc/clang with -shared and -fPIC)
    echo "Compiling as shared library..."
    clang $CFLAGS $PARALLEL_FLAG $DIAGNOSTIC_FLAG -fPIC -shared $SRC_FILES -o $OUTPUT_FILE
    if [[ $? -ne 0 ]]; then
        echo "Compilation of shared library failed. Exiting."
        exit 1
    fi

    echo "Shared library compiled successfully: $OUTPUT_FILE"
    exit 0
fi

# Create build directory if it doesn't exist
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Default Compilation (Executable)
echo "Compiling the program..."
clang $CFLAGS $PARALLEL_FLAG $DIAGNOSTIC_FLAG $SRC_FILES -o $OUTPUT_FILE

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
