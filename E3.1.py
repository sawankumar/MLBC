import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Define column names for your dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# Load the dataset
pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

# Display the first few rows of the dataset
pima.head()

# Define the features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]  # Features
y = pima.label  # Target variable

# Split the dataset into training and testing sets (70% training and 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create a Decision Tree Classifier object
clf = DecisionTreeClassifier()

# Train the Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Calculate and print the accuracy of the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
