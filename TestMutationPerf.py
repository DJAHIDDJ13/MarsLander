import time
import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Initialize the weights and biases for the hidden layers and output layer
        self.weights_input_hidden1 = np.random.randn(self.input_size, self.hidden_size1)
        self.biases_hidden1 = np.random.randn(self.hidden_size1)
        self.weights_hidden1_hidden2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.biases_hidden2 = np.random.randn(self.hidden_size2)
        self.weights_hidden2_output = np.random.randn(self.hidden_size2, self.output_size)
        self.biases_output = np.random.randn(self.output_size)
        
    def forward(self, input):
        # Propagate the input through the first hidden layer using the weights and biases
        hidden1 = np.dot(input, self.weights_input_hidden1) + self.biases_hidden1
        # Apply the ReLU activation function to the first hidden layer output
        hidden1 = np.maximum(hidden1, 0)
        
        # Propagate the output of the first hidden layer through the second hidden layer using the weights and biases
        hidden2 = np.dot(hidden1, self.weights_hidden1_hidden2) + self.biases_hidden2
        # Apply the ReLU activation function to the second hidden layer output
        hidden2 = np.maximum(hidden2, 0)

        # Propagate the output of the second hidden layer through the output layer using the weights and biases
        output = np.dot(hidden2, self.weights_hidden2_output) + self.biases_output

        return output
# Initialize a neural network with large number of weights
nn = NeuralNetwork(input_size=100, hidden_size1=100, hidden_size2=100, output_size=100)

# Test the performance of the original mutation method
mutation_probability = .1
start_time = time.time()
for _ in range(100):
    for i in range(nn.weights_input_hidden1.size):
        if np.random.uniform(0, 1) < mutation_probability:
            nn.weights_input_hidden1.flat[i] += np.random.normal(0, 1)
print("Original mutation method took {:.2f} seconds".format(time.time() - start_time))

# Test the performance of the new mutation method
nn = NeuralNetwork(input_size=100, hidden_size1=100, hidden_size2=100, output_size=100)
start_time = time.time()
for i in range(100):
    mutation_mask = np.random.binomial(1, mutation_probability, size=nn.weights_input_hidden1.shape)
    if i == 0:
        print(nn.weights_input_hidden1.shape)
    random_gaussian_noise = np.random.normal(0, 1, size=nn.weights_input_hidden1.shape) * 0.1
    nn.weights_input_hidden1 += mutation_mask * random_gaussian_noise
print("New mutation method took {:.2f} seconds".format(time.time() - start_time))
