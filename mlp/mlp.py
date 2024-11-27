import numpy as np

class MLP:
    def __init__(self, num_inputs, num_hidden, num_outputs, 
                 hidden_activation='sigmoid', output_activation='sigmoid'):
        """
        Initialize the MLP
        Args:
            num_inputs (int): Number of input features.
            num_hidden (int): Number of hidden units.
            num_outputs (int): Number of output units.
            hidden_activation (str): Activation function for hidden layer ('sigmoid', 'tanh', etc.).
            output_activation (str): Activation function for output layer ('sigmoid', 'linear', etc.).
        """
        self.error_history = [] # to plot the error history

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Initialize weights and biases
        self.W1 = np.random.uniform(-0.5, 0.5, (num_hidden, num_inputs))
        self.b1 = np.random.uniform(-0.5, 0.5, (num_hidden, 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (num_outputs, num_hidden))
        self.b2 = np.random.uniform(-0.5, 0.5, (num_outputs, 1))

        # Set activation functions | Lowercase for case-insensitivity
        self.hidden_activation = hidden_activation.lower()
        self.output_activation = output_activation.lower()

        # Define activation functions and their derivatives
        self.activation_funcs = {
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'tanh': (self.tanh, self.tanh_derivative),
            'linear': (self.linear, self.linear_derivative)
        }

        # Verify activation functions
        if self.hidden_activation not in self.activation_funcs:
            raise ValueError(f"Unsupported hidden activation function: {self.hidden_activation}")
        if self.output_activation not in self.activation_funcs:
            raise ValueError(f"Unsupported output activation function: {self.output_activation}")

    # Activation functions and their derivatives
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)

    def tanh_derivative(self, x):
        """Derivative of the tanh function."""
        return 1 - np.tanh(x) ** 2

    def linear(self, x):
        """Linear activation function."""
        return x

    def linear_derivative(self, x):
        """Derivative of the linear function."""
        return np.ones_like(x)
    # --- --- --- --- --- --- --- --- --- --- --- ---

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
        activation_hidden, _ = self.activation_funcs[self.hidden_activation]
        self.a1 = activation_hidden(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        activation_output, _ = self.activation_funcs[self.output_activation]
        self.output = activation_output(self.z2)
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

        # Derivative of output activation
        _, activation_output_derivative = self.activation_funcs[self.output_activation]
        output_delta = output_error * activation_output_derivative(self.z2)

        # Backpropagate to hidden layer
        hidden_error = np.dot(self.W2.T, output_delta)

        # Derivative of hidden activation
        _, activation_hidden_derivative = self.activation_funcs[self.hidden_activation]
        hidden_delta = hidden_error * activation_hidden_derivative(self.z1)

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
            self.error_history.append(total_error)
            # Print error every 1000 epochs
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}, Error: {total_error}")

    def evaluate(self, test_data, test_targets):
        total_error = 0
        correct_predictions = 0
        for input_vector, target in zip(test_data, test_targets):
            output = self.forward(input_vector)
            total_error += np.mean((output - target) ** 2)
            if self.num_outputs > 1:
                predicted = np.argmax(output)
                actual = np.argmax(target)
                if predicted == actual:
                    correct_predictions += 1
            else:
                # For regression tasks, define a threshold for 'correct' prediction
                # Here, we can consider predictions within a small epsilon as correct
                epsilon = 0.1
                if np.abs(output - target) < epsilon:
                    correct_predictions += 1
        average_error = total_error / len(test_data)
        if self.num_outputs > 1:
            accuracy = correct_predictions / len(test_data)
        else:
            accuracy = correct_predictions / len(test_data)  # Percentage of predictions within epsilon
        return average_error, accuracy