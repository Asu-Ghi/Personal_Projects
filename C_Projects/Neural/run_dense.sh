gcc -g -fsanitize=address -o main main.c network.c forward.c backward.c utils.c
$echo ./main 5 150 10 1.0