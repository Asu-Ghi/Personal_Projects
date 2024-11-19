gcc -shared -o NeuralNet.so -fPIC -arch x86_64 network.c forward.c backward.c utils.c test_functions.c
