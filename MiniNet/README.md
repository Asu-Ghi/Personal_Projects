---
title: "MiniNet"
author: "Asu Ghimire"
date: "2024-11"
output: html_document
---

## Current Work  
- Open MP support
- Visualization matplotlib toolkit in python

### Figure 1: Mnist Training Demonstration

![Mnist Demonstration](results/demonstrations/mnist_2_layer_dem.png)

### Figure 2: Spiral Strong Scaling Study
![Strong Scaling, Small NN](results/demonstrations/strong_scaling_plot_2.png)

### Figure 3: Mnist Strong Scaling Study
![Strong Scaling, Larger NN](results/demonstrations/strong_scaling_plot_Mnist1.png)

## Planned Work
- Support for CNN and RNN layers
- CUDA implementation for computationally heavier training tasks

## Problems
- Too 'javanic', cool down on the oop format.
- Doesnt make best use of cpu buffer and cache
- Not as efficient as id like, *see strong scaling studies*, Note that the larger the matrix math requirments are the better the parallelization scales with cpu cores.
