"""
Description:
    MLP with categorical cross entropy cost function for classifying MNIST images.
"""

import numpy as np

""" Defining Numpy Vector Functions"""
def sigmoid(x):
    """ Sigmoid/Logistic Function"""
    return 1 / (1 + np.exp(x))

def sigmoid_derivative(x):
    """ Derivative of Sigmoid Function"""
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x: np.ndarray):
    """ Softmax function for output layer """
    # # Values shifted to reduce chance of generating NaN values
    # x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def softmax_derivative(S: np.ndarray):
    """ Computes Gradient of Softmax Function """
    # Input S should be the vector output of the Softmax function
    S_diag = np.diag(S)
    S_vector = S.reshape(S.shape[0], 1)
    S_matrix = np.tile(S_vector, S.shape[0])

    # Compute jacobian derivative with matrix maths
    gradient = S_diag - (S_matrix * np.transpose(S_matrix))
    return gradient

def categorical_cross_entropy(predicted: np.ndarray, target: np.ndarray):
    """ Categorical Cross Entropy Cost Function"""
    losses = []
    for t, p in zip(target, predicted):
        loss = -np.sum(t * np.log(p))
        losses.append(loss)
    return losses

""" Defining Classes for MLP and Individual Layers """
class MLP:
    BATCH_SIZE = 32

    def __init__(self, num_inputs: int, num_outputs: int, hidden_layers_sizes: list):
        # Create Layers
        input_layer = Layer(num_inputs)
        output_layer = Layer(num_outputs)
        hidden_layers = [Layer(layer_size) for layer_size in hidden_layers_sizes]
        
        # Link Layers to form Sequential Graph
        self.graph = [input_layer] + hidden_layers + [output_layer]

        # Initialise Weights and Biases
        self.initialise_weights()
        self.initialise_biases()

    def initialise_weights(self):
        """ Create weight vectors with random values between 0 and 1 """
        for i in range(1, len(self.graph)):
            layer = self.graph[i]
            prev_layer_size = self.graph[i-1].get_layer_size()
            layer.initialise_weights(prev_layer_size)

    def initialise_biases(self):
        """ Create bias vectors with random values between 0 and 1 """
        for i in range(1, len(self.graph)):
            layer: Layer = self.graph[i]
            layer.initialise_biases()

    def train(self, x_data, y_data, epochs: int = 3):
        pass

    def print_graph(self):
        for i, layer in enumerate(self.graph):
            print("---------------------------------------------")
            print(f"---------------- Layer {i} -----------------")
            print(f"Neurons: {layer.get_layer_size()}")
            if layer.get_bias_vector() is not None and layer.get_weight_vectors is not None:
                print(f"Biases: {layer.get_bias_vector().size}")
                print(f"Weights: {layer.get_weight_vectors().shape}")
            else:
                print(f"Biases: {layer.get_bias_vector()}")
                print(f"Weights: {layer.get_weight_vectors()}")

        print("---------------------------------------------")

class Layer:
    def __init__(self, layer_size: int):
        # Record number of neurons in layer
        self.layer_size = layer_size
        # Create activations which stores neuron outputs
        # Create bias and weight attributes to initialise later
        self.activation_values = None
        self.biases = None
        self.weights = None

    def initialise_weights(self, prev_layer_size: int):
        """ Initialise one weight value for each neuron connection"""
        self.weights = np.random.rand(self.layer_size, prev_layer_size)
    
    def initialise_biases(self):
        """ Initialise bias value for each neuron in layer"""
        self.biases = np.random.rand(self.layer_size)

    def get_layer_size(self):
        return self.layer_size
    
    def get_weight_vectors(self):
        return self.weights
    
    def get_bias_vector(self):
        return self.biases
