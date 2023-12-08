"""
Description:
    MLP with categorical cross entropy cost function for classifying MNIST images.
"""

import numpy as np

""" Defining Numpy Vector Functions """
def sigmoid(x):
    """ Sigmoid/Logistic Function """
    return 1 / (1 + np.exp(x))

def sigmoid_derivative(x):
    """ Derivative of Sigmoid Function """
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x: np.ndarray):
    """ Softmax function for Output Vectors """
    # # Values shifted to reduce chance of generating NaN values
    # x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(S: np.ndarray):
    """ Computes Gradient of Softmax Function """
    # Input S should be the vector output of the Softmax function
    gradients = []
    for row in S:
        S_diag = np.diag(row)
        S_vector = row.reshape(row.shape[0], 1)
        S_matrix = np.tile(S_vector, row.shape[0])

        # Compute jacobian derivative with matrix maths
        gradient = S_diag - (S_matrix * np.transpose(S_matrix))
        gradients.append(gradient)
    return np.stack(gradients, axis=0)

def categorical_cross_entropy(predictions: np.ndarray, targets: np.ndarray):
    """ Categorical Cross Entropy Cost Function for a Batch of Samples """
    losses = []
    for t, p in zip(targets, predictions):
        loss = -np.sum(t * np.log(p))
        losses.append(loss)
    return np.sum(losses)

def cce_derivative(predictions: np.ndarray, targets: np.ndarray):
    """ Derivative of Categorical Cross Entropy cost function """
    # Gradient is simply the difference between the
    # predicted probability values and the true probability values.
    return predictions - targets

""" Defining Classes for MLP and Individual Layers """
class MLP:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.1

    def __init__(self, num_inputs: int, num_outputs: int, hidden_layers_sizes: list):
        # Create Layers
        input_layer = Layer(num_inputs)
        output_layer = Layer(num_outputs, "Softmax")
        hidden_layers = [Layer(layer_size, "Sigmoid") for layer_size in hidden_layers_sizes]
        
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

    def train(self, x_data: np.ndarray, y_data: np.ndarray, epochs: int = 3):
        """ Train the network """
        sample_count = x_data.shape[0]
        print(f"Training network on {sample_count} examples for {epochs} epochs.")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        for epoch in range(epochs):
            # Slice data into batches for training and keep going until all data is used
            batch_start_index = 0
            batch_end_index = self.BATCH_SIZE
            while batch_end_index < sample_count:
                # Create slices
                batch_x = x_data[batch_start_index:batch_end_index]
                batch_y = y_data[batch_start_index:batch_end_index]
                batch_start_index = batch_end_index
                batch_end_index += self.BATCH_SIZE
                # Feed forward to get predictions
                batch_y_hat = self.feed_forward(batch_x)
                print(batch_y_hat.shape, batch_y.shape)
                prev_error: np.ndarray = batch_y_hat - batch_y
                # Iterate backwards through graph and find gradients
                for i in reversed(range(1, len(self.graph))):
                    layer: Layer = self.graph[i]
                    layer.weight_grad = prev_error.dot(self.graph[i-1].activation_values)
                    layer.bias_grad = prev_error
                    activation_error = layer.activation_function_derivative(layer.pre_activation_values)
                    print(layer.pre_activation_values.shape)
                    break
                    # error_chain = np.sum()
                    # layer.weight_grad = error_chain.dot()

                # # Update weights and biases according to gradients
                # for i in reversed(range(1, len(self.graph))):
                #     layer: Layer = self.graph[i]
                #     layer.update_weights(self.LEARNING_RATE)
                #     layer.update_biases(self.LEARNING_RATE)
                break
            # Use categorical cross entropy function to get loss
            loss = categorical_cross_entropy(batch_y_hat, batch_y)
            print(f"Epoch {epoch+1} completed. Loss: {loss}")

    def feed_forward(self, x_data: np.ndarray):
        """ Run input through network and return prediction """
        prev_output = x_data.reshape((self.BATCH_SIZE, -1))
        self.graph[0].activation_values = prev_output
        for i in range(1, len(self.graph)):
            layer: Layer = self.graph[i]
            layer.pre_activation_values = prev_output.dot(layer.weights) + layer.biases
            layer.activation_values = layer.activation_function(layer.pre_activation_values)
            prev_output = layer.activation_values
        return self.graph[-1].activation_values

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
    def __init__(self, layer_size: int, activation_function=None):
        # Record number of neurons in layer
        self.layer_size = layer_size
        # Create activations which stores neuron outputs
        # Create bias and weight attributes to initialise later
        self.activation_values = None
        self.pre_activation_values = None
        self.biases = None
        self.weights = None
        self.bias_grad = None
        self.weight_grad = None
        # Set activation function
        if activation_function == "Sigmoid":
            self.activation_function = sigmoid
            self.activation_function_derivative = sigmoid_derivative
        elif activation_function == "Softmax":
            self.activation_function = softmax
            self.activation_function_derivative = softmax_derivative
        else:
            self.activation_function = None
            self.activation_function_derivative = None

    def initialise_weights(self, prev_layer_size: int):
        """ Initialise one weight value for each neuron connection"""
        self.weights = np.random.rand(prev_layer_size, self.layer_size)
    
    def initialise_biases(self):
        """ Initialise bias value for each neuron in layer"""
        self.biases = np.random.rand(self.layer_size)

    def update_weights(self, learning_rate):
        """ Update existing weights according to gradient """
        self.weights -= learning_rate * self.weight_grad

    def update_biases(self, learning_rate):
        """ Update existing biases according to gradient """
        self.biases -= learning_rate * self.bias_grad

    def get_layer_size(self):
        return self.layer_size
    
    def get_weight_vectors(self):
        return self.weights
    
    def get_bias_vector(self):
        return self.biases
