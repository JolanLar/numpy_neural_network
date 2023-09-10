import numpy as np

np.random.seed(1)  # Seed the random number generator

class NeuronLayer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        # Initialize weights with small random values
        self.synaptic_weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

class NeuralNetwork:
    def __init__(self, layer1, layer2, layer3, learning_rate=0.01):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output_from_layer_1, output_from_layer_2, output_from_layer_3 = self.think(training_set_inputs)

            layer3_error = training_set_outputs - output_from_layer_3
            layer3_delta = layer3_error * self.sigmoid_derivative(output_from_layer_3)

            layer2_error = layer3_delta.dot(self.layer3.synaptic_weights.T)
            layer2_delta = layer2_error * self.sigmoid_derivative(output_from_layer_2)

            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.sigmoid_derivative(output_from_layer_1)

            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)

            # Update weights with learning rate
            self.layer1.synaptic_weights += self.learning_rate * layer1_adjustment
            self.layer2.synaptic_weights += self.learning_rate * layer2_adjustment
            self.layer3.synaptic_weights += self.learning_rate * layer3_adjustment

            if iteration % 1000 == 0:
                print(f"Iteration {iteration}: Error {np.mean(np.abs(layer3_error))}")

    def think(self, inputs):
        output_from_layer1 = self.sigmoid(np.dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.sigmoid(np.dot(output_from_layer1, self.layer2.synaptic_weights))
        output_from_layer3 = self.sigmoid(np.dot(output_from_layer2, self.layer3.synaptic_weights))
        return output_from_layer1, output_from_layer2, output_from_layer3

if __name__ == "__main__":
    # Create layer 1 (200 neurons, each with 3 inputs)
    layer1 = NeuronLayer(200, 3)

    # Create layer 2 (200 neurons, each with 200 inputs)
    layer2 = NeuronLayer(200, 200)

    # Create layer 3 (3 neurons, each with 200 inputs)
    layer3 = NeuronLayer(3, 200)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2, layer3, learning_rate=0.01)

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 3 output values.
    training_set_inputs = np.array([[0.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0]])

    # The training set outputs should be of float64 type
    training_set_outputs = np.array([[1.0, 1.0, 0.0],
                                     [1.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [1.0, 0.0, 1.0],
                                     [0.0, 1.0, 1.0],
                                     [0.0, 0.0, 0.0],
                                     [1.0, 1.0, 1.0]])

    # Normalize inputs
    training_set_inputs /= np.max(training_set_inputs)

    # Train the neural network using the training set.
    # Do it 120,000 times to allow for better convergence.
    neural_network.train(training_set_inputs, training_set_outputs, 120000)

    # Iterate through all possible combinations of [0, 1] for the input
    input_combinations = [(i, j, k) for i in [0, 1] for j in [0, 1] for k in [0, 1]]

    for input_combination in input_combinations:
        input_data = np.array(input_combination)
        print(f"Considering a new situation {input_data} -> ?:")
        _, _, output = neural_network.think(input_data)
        rounded_output = np.round(output).astype(int)  # Round to integers
        print(rounded_output)