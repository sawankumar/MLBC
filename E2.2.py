# Make sure you have already installed the required libraries using pip.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Import the dataset
try:
    dataset = pd.read_csv('LogisticRegressiondata.csv')
except FileNotFoundError:
    print("Error: The file 'LogisticRegressiondata.csv' not found.")
    exit()

# Print the dataset
print(dataset)

# Split the data into inputs and outputs
X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 2].values

# Count the total output data from the 'Purchased' column
target_balance = dataset['Purchased'].value_counts().reset_index()

# Create variables for class counts
class_one = 0
class_two = 0

# Iterate through the output class to count occurrences
for i in y:
    if i == 0:
        class_one += 1
    else:
        class_two += 1

# Create numpy array for pie chart
values = np.array([class_one, class_two])
label = ["Not-Purchased", "Purchased"]

# Plot the pie chart
plt.pie(values, labels=label)
plt.show()

# Print the results
print("Not purchased:", class_one)
print("Purchased:", class_two)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Independent class:\n", X_train[:10])
