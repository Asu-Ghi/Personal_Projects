#!/bin/bash

# Set stack size limit
ulimit -s 16384  # For example, 16 MB stack size

# Set OpenMP environment variables
export OMP_DEBUG=true
export OMP_NUM_THREADS=6
export OMP_DISPLAY_ENV=true
export OMP_SCHEDULE=DYNAMIC
export OMP_PROC_BIND=TRUE

# Project directories
SRC_DIR="src"
BUILD_DIR="build"
PYTHON_DIR="python"
INCLUDE_DIR="include"

# Default Flags
PARALLEL_FLAG=""
DEBUG_FLAG=""
SHARED_FLAG=""
OUTPUT_FILE=""

# Check for the command-line argument for parallel flag
if [[ "$1" == "-parallel" ]]; then
    PARALLEL_FLAG="-DENABLE_PARALLEL"
    echo "Compiling with parallelization enabled..."
else
    echo "Compiling without parallelization..."
fi

# Check for debugging flag 
if [[ "$2" == "-diag" ]]; then
    DEBUG_FLAG="lldb"
    echo "Debugging with lldb enabled..."
fi

# Check for the shared library flag
if [[ "$3" == "-shared" ]]; then
    SHARED_FLAG="-shared -fPIC"
    OUTPUT_FILE="${BUILD_DIR}/libnn.so"
    echo "Compiling as a shared library..."
else
    OUTPUT_FILE="${BUILD_DIR}/main"
    echo "Compiling as an executable..."
fi

# Verify build directory
mkdir -p "$BUILD_DIR"

# Get source files
SRC_FILES=$(find "$SRC_DIR" -name '*.c' -print | tr '\n' ' ')

# Compile with flags
if [[ -n "$SHARED_FLAG" ]]; then
    # Compile as shared library
    gcc-14 -O3 -arch x86_64 $PARALLEL_FLAG -march=native -funroll-loops -ftree-vectorize -fopenmp \
        -I"$INCLUDE_DIR" $SHARED_FLAG -o "$OUTPUT_FILE" $SRC_FILES
else
    # Compile as executable
    gcc-14 -O3 $PARALLEL_FLAG -march=native -funroll-loops -ftree-vectorize -fopenmp \
        -I"$INCLUDE_DIR" -o "$OUTPUT_FILE" $SRC_FILES
fi

# Check compilation 
if [ $? -eq 0 ]; then
    echo "Compilation successful."

else
    echo "Error compiling C code."
    exit 1
fi

# Clean up compiled objects and executable or shared library
echo "Cleaning up compiled objects..."
make clean
