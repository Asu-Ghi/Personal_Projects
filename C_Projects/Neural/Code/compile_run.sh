# Compile 
# gcc -o main main.c forward.c backward.c utils.c network.c test_functions.c
gcc -g -fsanitize=address -o main main.c network.c forward.c backward.c utils.c test_functions.c
# Run
./main 