import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target
y = pd.get_dummies(y).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)

learning_rate = 0.1
iterations = 5000
N = y_train.shape[0]
input_size = 4
hidden_size = 2
output_size = 3

results = pd.DataFrame(columns=["mse", "accuracy"])
np.random.seed(10)

W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))
W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def accuracy(y_pred, y_true):
    correct = (np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)).sum()
    return correct / len(y_pred)

mse_values = []

for itr in range(iterations):
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    mse = mean_squared_error(A2, y_train)
    mse_values.append(mse)  # Store the MSE values

    acc = accuracy(A2, y_train)
    results = pd.concat([results, pd.DataFrame({"mse": [mse], "accuracy": [acc]})], ignore_index=True)

    E2 = y_train - A2
    dW2 = E2 * sigmoid_derivative(A2)

    E1 = dW2.dot(W2.T)
    dW1 = E1 * sigmoid_derivative(A1)

    W2 += A1.T.dot(dW2) * learning_rate
    W1 += X_train.T.dot(dW1) * learning_rate

# Plot the MSE values
plt.figure(figsize=(10, 5))
plt.plot(range(iterations), mse_values, label="MSE")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.title("Mean Squared Error (MSE) over Iterations")
plt.legend()
plt.show()

# Plot the accuracy values
plt.figure(figsize=(10, 5))
results.accuracy.plot(title="Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Accuracy over Iterations")
plt.show()
