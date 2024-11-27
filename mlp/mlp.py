import numpy as np

class MLP:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        """
        Initialize the MLP with given structure.
        Args:
            num_inputs (int): Number of input features.
            num_hidden (int): Number of hidden units.
            num_outputs (int): Number of output units.
        """
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Initialize weights and biases
        self.W1 = np.random.uniform(-0.5, 0.5, (num_hidden, num_inputs))
        self.b1 = np.random.uniform(-0.5, 0.5, (num_hidden, 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (num_outputs, num_hidden))
        self.b2 = np.random.uniform(-0.5, 0.5, (num_outputs, 1))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, input_vector):
        """
        Perform a forward pass.
        Args:
            input_vector (array-like): Input to the network.
        Returns:
            array-like: Output of the network.
        """
        self.input = np.array(input_vector).reshape(self.num_inputs, 1)
        self.z1 = np.dot(self.W1, self.input) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.output = self.sigmoid(self.z2)  # Assuming sigmoid output
        return self.output

    def backward(self, target_vector, learning_rate):
        """
        Perform a backward pass and update weights.
        Args:
            target_vector (array-like): The expected output.
            learning_rate (float): Learning rate for weight updates.
        Returns:
            float: Mean squared error for this example.
        """
        target = np.array(target_vector).reshape(self.num_outputs, 1)
        output_error = self.output - target
        output_delta = output_error * self.sigmoid_derivative(self.z2)

        hidden_error = np.dot(self.W2.T, output_delta)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.z1)

        # Update weights and biases
        self.W2 -= learning_rate * np.dot(output_delta, self.a1.T)
        self.b2 -= learning_rate * output_delta
        self.W1 -= learning_rate * np.dot(hidden_delta, self.input.T)
        self.b1 -= learning_rate * hidden_delta

        # Compute and return mean squared error
        return np.mean(output_error**2)
    
    def train(self, training_data, targets, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            for input_vector, target in zip(training_data, targets):
                self.forward(input_vector)
                error = self.backward(target, learning_rate)
                total_error += error
            # Added print every 1000, as printing every epoch seems excessive in this scenario
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}, Error: {total_error}")

    def evaluate(self, test_data, test_targets):
        total_error = 0
        correct_predictions = 0
        for input_vector, target in zip(test_data, test_targets):
            output = self.forward(input_vector)
            total_error += np.mean((output - target) ** 2)
            # Use self.num_outputs instead of self.NO
            predicted = np.argmax(output) if self.num_outputs > 1 else output
            actual = np.argmax(target) if self.num_outputs > 1 else target
            if np.argmax(output) == np.argmax(target) if self.num_outputs > 1 else output == target:
                correct_predictions += 1
        average_error = total_error / len(test_data)
        accuracy = correct_predictions / len(test_data)
        return average_error, accuracy




