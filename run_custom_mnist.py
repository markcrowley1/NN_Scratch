
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from custom_nn import *

def view_data(x):
    print(x.shape)
    plt.imshow(x, cmap = plt.cm.binary)
    plt.show()
    return

def main():
    # Load in MNIST data
    dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # Normalise Inputs for Training and Test Data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    image, label = x_train[0], y_train[0]

    # Convert labels to one hot encoding
    y_train_oh = np.zeros((y_train.size, y_train.max() + 1))
    y_train_oh[np.arange(y_train.size), y_train] = 1

    # Create network and print graph
    model = MLP(784, 10, [128,128])
    # predictions = model.feed_forward(image)
    # print(np.sum(predictions), predictions)
    model.train(x_train, y_train_oh, epochs=1)
    # view_data(image)

    # # Test out activation functions
    # # Generate random values to represent NN output
    # pseudo_output = np.random.rand(5, 10)
    # pseudo_target = np.tile([1,0,0,0,0,0,0,0,0,0], (5,1))
    # print(f"Predictions:\n {pseudo_output}\n")
    # print(f"Targets:\n {pseudo_target}\n")
    # # Check the output of the Softmax function
    # S = softmax(pseudo_output)
    # print(f"{np.sum(S)}, {S}")
    # # Test out categorical cross entropy loss function
    # loss = categorical_cross_entropy(S, pseudo_target)
    # print(f"Loss:\n {loss}\n")
    # dL_dz = cce_derivative(S, pseudo_target)
    # print(f"Derivative of Cost Function:\n {dL_dz}\n")
    # # # Find the gradient of the softmax
    # # dS_dx = softmax_derivative(S)
    
    # # Test out sigmoid function and its derivative
    # A = sigmoid(pseudo_output)
    # print(pseudo_output)
    # dA_dx = sigmoid_derivative(A)
    # print(dA_dx)


if __name__ == "__main__":
    main()