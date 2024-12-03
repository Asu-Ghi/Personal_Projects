---
title: "MiniNet"
author: "Asu Ghimire"
date: "2024-11"
output: html_document
---
## Current Work  
- Starting Cuda Support
- Starting CNN Support
- Completing Mini Batch Support
- Completing Open MP support
- Developing Visualization matplotlib toolkit in python

## Planned Work
- Support for CNN and RNN layers
- CUDA implementation for computationally heavier training tasks

## Problems
- Too 'javanic', cool down on the oop format.
- Doesnt make best use of cpu buffer and cache
- Not as efficient as id like, *see strong scaling studies*

## Results
### Figure 1: Mnist Training Demonstration with 3 layers (Full Batch)
- Full batch training results in longer training times and greater number of epochs required to reach satisfactory validation accuracy.
![Mnist Demonstration 3 Layers](results/demonstrations/Mnist_3_layer_dem.png)

### Figure 2: Mnist Strong Scaling Study (Full Batch)
- Underperforms on full batch training, parallelization not fully utilized. 
![Strong Scaling, Mnist Full Batch](results/demonstrations/strong_scaling_plot_Mnist1.png)

### Figure 3: Mnist Training Demonstration with 4 layers (Mini Batch)
- Performs noticeably faster than 3 layered full batch training, achieves target accuracy within 15 epochs. Full batch size of 10k, with mini batches of size 1k samples. 
![Mnist Demonstration 4 Layers](results/demonstrations/mnist_4_layer_minibatch_dem.png)

### Figure 4: Mnist Strong Scaling Study (Mini Batch)
- Parallelization performs slightly better on mini batch training, the lack of performance on 8 cores might be due to
the limiting nature of omp not allowing me to fully utilize all 8 cores on my current machine.
![Strong Scaling, MNIST Mini Batch](results/demonstrations/strong_scaling_plot_Mnist_minibatch.png)

