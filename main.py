from mlp.mlp import MLP
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

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

# to plot error history
def plot_error(mlp, title):
    plt.figure()
    plt.plot(mlp.error_history)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Total Error')
    plt.grid(True)
    plt.show()
# --- --- --- --- --- --- --- --- ---
# mlp training for XOR related ---
def train_xor():
    training_data, targets = get_xor_data()
    
    # Initialize MLP
    # For classification with sigmoid activation in output
    mlp = MLP(num_inputs=2, num_hidden=4, num_outputs=1, 
              hidden_activation='sigmoid', output_activation='sigmoid')
    
    # Training params
    epochs = 10000  # Adjust as needed, Experimented with 100,000 epochs -> signifcantly longer training times, improved results. Priorities time or accuracy for epoch number
    learning_rate = 0.1
    
    # Train
    print("Training MLP on XOR data...")
    mlp.train(training_data, targets, epochs, learning_rate)
    plot_error(mlp, "XOR Training Error Over Epochs")
    
    return mlp, training_data, targets

def test_xor(mlp, training_data, targets):
    print("\nTesting the MLP on XOR data:")
    for input_vector, target in zip(training_data, targets):
        output = mlp.forward(input_vector)
        if mlp.num_outputs > 1:
            predicted = np.argmax(output)
        else:
            predicted = np.round(output.flatten(), 3)  # Round for readability
        print(f"Input: {input_vector}, Predicted: {predicted}, Target: {target}")

def train_test_mlp_for_xor(): # making code more modular so main function is not too messy or complicated!
    # XOR Classification Task
    print("=== XOR Classification ===")
    mlp_xor, training_data_xor, targets_xor = train_xor()
    test_xor(mlp_xor, training_data_xor, targets_xor)
# --- --- --- --- --- --- --- --- ---
# mlp training for sinusoidal data realted ---
def get_sinusoidal_data(num_samples=500):
    input_vectors = [np.random.uniform(-1, 1, 4) for _ in range(num_samples)]
    output_vectors = [[math.sin(vec[0] - vec[1] + vec[2] - vec[3])] for vec in input_vectors]
    return input_vectors, output_vectors

def split_data(input_vectors, output_vectors, train_ratio=0.8):
    split_index = int(len(input_vectors) * train_ratio)
    train_data = input_vectors[:split_index]
    train_targets = output_vectors[:split_index]
    test_data = input_vectors[split_index:]
    test_targets = output_vectors[split_index:]
    return train_data, train_targets, test_data, test_targets

def train_sinusoidal():
    # Generate data
    input_vectors, output_vectors = get_sinusoidal_data()
    
    # Split into training and testing
    train_data, train_targets, test_data, test_targets = split_data(input_vectors, output_vectors)
    
    # Initialize MLP
    # For regression with linear activation in output
    mlp = MLP(num_inputs=4, num_hidden=5, num_outputs=1, 
              hidden_activation='sigmoid', output_activation='linear')
    
    # Training params
    epochs = 1000
    learning_rate = 0.01
    
    # Train the MLP
    print("\nTraining MLP on Sinusoidal data...")
    mlp.train(train_data, train_targets, epochs, learning_rate)
    plot_error(mlp, "Sinusoidal Training Error Over Epochs")
    
    # Evaluate on training data
    train_error, _ = mlp.evaluate(train_data, train_targets)
    print(f"Training Error (MSE): {train_error}")
    
    # Evaluate on testing data
    test_error, _ = mlp.evaluate(test_data, test_targets)
    print(f"Testing Error (MSE): {test_error}")
    
    return mlp, train_data, train_targets, test_data, test_targets
# --- --- --- --- --- --- --- --- ---

def main():
    parser = argparse.ArgumentParser(description="MLP Assignment")
    parser.add_argument('--task', type=str, choices=['xor', 'sinusoidal', 'all'], default='all',
                        help='Task to perform: xor, sinusoidal, or all')

    args = parser.parse_args()

    if args.task in ['xor', 'all']:
        train_test_mlp_for_xor()
    
    if args.task in ['sinusoidal', 'all']:
        print("\n=== Sinusoidal Regression ===")
        mlp_sin, train_data_sin, train_targets_sin, test_data_sin, test_targets_sin = train_sinusoidal()


    
if __name__ == "__main__":
    main()

