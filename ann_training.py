import numpy as np

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# Network dimensions
input_size = 4
hidden_size = 3
output_size = 2
learning_rate = 0.5
epochs = 2

# Input and target output
X = np.array([[1, 0, 1, 0]])  # Shape: (1, 4)
Y = np.array([[1, 0]])        # Shape: (1, 2)

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.rand(input_size, hidden_size)   # (4x3)
b1 = np.random.rand(1, hidden_size)            # (1x3)
W2 = np.random.rand(hidden_size, output_size)  # (3x2)
b2 = np.random.rand(1, output_size)            # (1x2)

# Train for 2 epochs
for epoch in range(epochs):
    # === Forward Pass ===
    z1 = np.dot(X, W1) + b1     # (1x3)
    a1 = sigmoid(z1)            # (1x3)
    z2 = np.dot(a1, W2) + b2    # (1x2)
    y_hat = sigmoid(z2)         # (1x2)

    # === Compute Loss ===
    loss = 0.5 * np.sum((Y - y_hat)**2)
    print(f"Epoch {epoch+1} - Loss: {loss:.4f}, Output: {y_hat}")

    # === Backward Pass ===
    delta_output = (Y - y_hat) * sigmoid_deriv(z2)             # (1x2)
    delta_hidden = sigmoid_deriv(z1) * np.dot(delta_output, W2.T)  # (1x3)

    # === Update Weights and Biases ===
    W2 += learning_rate * np.dot(a1.T, delta_output)           # (3x2)
    b2 += learning_rate * delta_output                         # (1x2)
    W1 += learning_rate * np.dot(X.T, delta_hidden)            # (4x3)
    b1 += learning_rate * delta_hidden                         # (1x3)