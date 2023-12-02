
import numpy as np

from custom_nn import *

def main():
    # Create network and print graph
    model = MLP(784, 10, [128,128])
    model.print_graph()

    # Test out activation functions
    # Generate random values to represent NN output
    pseudo_output = np.random.rand(10)
    print(pseudo_output)
    # Check the output of the Softmax function
    S = softmax(pseudo_output)
    print(np.sum(S), S)
    # Find the gradient of the softmax
    dS_dx = softmax_derivative(S)
    
    # Test out sigmoid function and its derivative
    A = sigmoid(pseudo_output)
    print(pseudo_output)
    dA_dx = sigmoid_derivative(A)
    print(dA_dx)


if __name__ == "__main__":
    main()