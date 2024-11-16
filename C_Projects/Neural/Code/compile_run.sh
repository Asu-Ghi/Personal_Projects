# Compile 
gcc -g -fsanitize=address -o main main.c network.c forward.c backward.c utils.c test_functions.c
# Run
./main 