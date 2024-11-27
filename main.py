from mlp.mlp import MLP

# Create an MLP with 2 inputs, 3 hidden units, and 1 output
mlp = MLP(num_inputs=2, num_hidden=3, num_outputs=1)

# XOR dataset
training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

# Training loop
epochs = 10000
learning_rate = 0.1
for epoch in range(epochs):
    total_error = 0
    for input_vector, target in training_data:
        mlp.forward(input_vector)
        total_error += mlp.backward(target, learning_rate)
    if epoch % 1000 == 0:  # Print error every 1000 epochs
        print(f"Epoch {epoch}, Error: {total_error}")

# Test the trained MLP
print("\nTesting the MLP:")
for input_vector, target in training_data:
    output = mlp.forward(input_vector)
    print(f"Input: {input_vector}, Predicted: {output.flatten()}, Target: {target}")
