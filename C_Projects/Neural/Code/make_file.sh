# Compile 

# gcc-14 -g -fopenmp -fsanitize=address -o main main.c test_functions.c network.c layer_dense.c utils.c
gcc-14 -fopenmp -o main main.c test_functions.c network.c layer_dense.c utils.c


# Check if compilation was successful (exit status 0 indicates success)
if [ $? -eq 0 ]; then
    # Run the compiled program if compilation was successful
    ./main
else
    echo "Compilation failed. Exiting."
fi