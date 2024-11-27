from mlp.mlp import MLP
import numpy as np
import math

# Create an MLP with 2 inputs, 3 hidden units, and 1 output
mlp = MLP(num_inputs=2, num_hidden=3, num_outputs=1)

def get_xor_data():
    # Inputs for simple binary XOR operations
    training_data = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    
    # XOR target outputs
    targets = [
        [0],
        [1],
        [1],
        [0]
    ]
    
    return training_data, targets

def train_xor():
    training_data, targets = get_xor_data()
    
    # Initialize MLP
    mlp = MLP(num_inputs=2, num_hidden=4, num_outputs=1)
    
    # Training params
    epochs = 10000 # 10,000 was relatively accurate, but 100,000 is slightly more accurate but takes significantly more training time.
    learning_rate = 0.1
    
    # Train
    mlp.train(training_data, targets, epochs, learning_rate)
    
    return mlp, training_data, targets

def test_xor(mlp, training_data, targets):
    print("\nTesting the MLP on XOR data:")
    for input_vector, target in zip(training_data, targets):
        output = mlp.forward(input_vector)
        predicted = np.round(output.flatten(), 3)  # Round for readability
        print(f"Input: {input_vector}, Predicted: {predicted}, Target: {target}")

def train_test_mlp_for_xor(): # making code more modular so main function is not too messy or complicated!
    # Train & Test the MLP on XOR data
    mlp, training_data, targets = train_xor()
    test_xor(mlp, training_data, targets)


def main():
    
    # XOR
    train_test_mlp_for_xor()

    


if __name__ == "__main__":
    main()

