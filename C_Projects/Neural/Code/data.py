import numpy as np


# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/
def create_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y



def save_data_to_csv(X, y, X_file="X_data.csv", y_file="y_labels.csv"):
    # Save features to CSV
    np.savetxt(X_file, X, delimiter=",")
    
    # Save labels to CSV
    np.savetxt(y_file, y, delimiter=",", fmt="%d")  # Save as integers

X, y = create_data(samples = 100, classes = 3)
save_data_to_csv(X, y)