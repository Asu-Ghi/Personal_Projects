#!/bin/bash

# Set stack size limit
ulimit -s 16384  # For example, 16 MB stack size

# Set OpenMP environment variables
export OMP_DEBUG=true
export OMP_NUM_THREADS=6
export OMP_DISPLAY_ENV=true
export OMP_SCHEDULE=DYNAMIC
export OMP_PROC_BIND=TRUE

# Default Flags
PARALLEL_FLAG=""
DEBUG_FLAG=""
SHARED_FLAG=""

# Check for the command-line argument for parallel execution
if [[ "$1" == "-parallel" ]]; then
    PARALLEL_FLAG="-D ENABLE_PARALLEL"
    echo "Compiling with parallelization enabled..."
else
    echo "Compiling without parallelization..."
fi

# Check if debugging flag is passed
if [[ "$2" == "-diag" ]]; then
    DEBUG_FLAG="lldb"
    echo "Debugging with lldb enabled..."
fi

# Check for the shared library flag
if [[ "$3" == "-shared" ]]; then
    SHARED_FLAG="-shared -fPIC"
    echo "Compiling as a shared library..."
else
    echo "Compiling as an executable..."
fi

# Compile with the appropriate flags
if [[ -n "$SHARED_FLAG" ]]; then
    # Compile as shared library
    gcc-14 -O3 $PARALLEL_FLAG -march=native -funroll-loops -ftree-vectorize -fopenmp $SHARED_FLAG -o libnn.so main.c test_functions.c network.c layer_dense.c utils.c
else
    # Compile as executable
    gcc-14 -O3 $PARALLEL_FLAG -march=native -funroll-loops -ftree-vectorize -fopenmp -o main main.c test_functions.c network.c layer_dense.c utils.c
fi

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful."

    # If debugging flag is set, use lldb; otherwise, run normally
    if [[ -n "$DEBUG_FLAG" ]]; then
        echo "Running with lldb debugger..."
        $DEBUG_FLAG ./main
    else
        # Run the Python script
        echo "Running the Python script..."
        python3 train_nn.py
    fi
else
    echo "Error compiling C code."
    exit 1
fi

# Clean up compiled objects and executable or shared library
echo "Cleaning up compiled objects..."
make clean
