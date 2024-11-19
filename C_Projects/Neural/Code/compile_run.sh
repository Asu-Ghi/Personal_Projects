# Compile 
# gcc -o main main.c forward.c backward.c utils.c network.c test_functions.c
gcc -g -fsanitize=address -o main main.c network.c forward.c backward.c utils.c test_functions.c

# Check if compilation was successful (exit status 0 indicates success)
if [ $? -eq 0 ]; then
    # Run the compiled program if compilation was successful
    ./main
else
    echo "Compilation failed. Exiting."
fi