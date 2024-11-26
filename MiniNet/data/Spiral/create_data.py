"""
Asu Ghimire
11/17/2024

Interface for working with Self Made C-Library for Neural Networks.

"""

import numpy as np

# Andrej Karpathy create data code
def create_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number

    # Randomize the data order
    indices = np.arange(X.shape[0])  # Create an array of indices
    np.random.shuffle(indices)  # Shuffle the indices
    X = X[indices]  # Reorder X
    y = y[indices]  # Reorder y

    return X, y

# Create one hot labels from spiral data
def create_one_hot(y, num_classes):
    """
    Convert class labels to one-hot encoded vectors.
    :param y: Array of class labels, shape (num_samples,)
    :param num_classes: Number of distinct classes
    :return: One-hot encoded matrix, shape (num_samples, num_classes)
    """
    return np.eye(num_classes)[y]

# Save data to csv 
def save_data_csv(X, y, x_filename="features.csv", y_filename="labels.csv"):
    """
    Save the feature matrix X and label array y to separate CSV files.
    
    :param X: Feature matrix of shape (num_samples, num_features)
    :param y: Label array of shape (num_samples,)
    :param x_filename: Filename for the feature matrix (default is "features.csv")
    :param y_filename: Filename for the label array (default is "labels.csv")
    """
    # Save X to CSV
    np.savetxt(x_filename, X, delimiter=',')
    print(f"Features saved to {x_filename}")
    
    # Save y to CSV
    np.savetxt(y_filename, y, delimiter=',')
    print(f"Labels saved to {y_filename}")

# main
def main():
    num_samples = 10000
    num_classes = 3

    # Generate data
    X, y = create_data(num_samples, num_classes)
    y = create_one_hot(y, num_classes)

    save_data_csv(X, y, "./test_data_10000.csv", "./test_labels_10000.csv")

if __name__ == "__main__":
    main()
