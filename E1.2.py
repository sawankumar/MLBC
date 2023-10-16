# First, make sure you have installed the necessary libraries
# You can use pip install or conda install depending on your environment
# pip install pandas
# pip install matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Corrected import statement for matplotlib

dataFrame = pd.read_csv('Age_Income.csv')
age = dataFrame['Age']
income = dataFrame['Income']
num = np.size(age)
mean_age = np.mean(age)
mean_income = np.mean(income)
CD_ageincome = np.sum(income * age) - num * mean_income * mean_age
CD_ageage = np.sum(age * age) - num * mean_age * mean_age
b1 = CD_ageincome / CD_ageage
b0 = mean_income - b1 * mean_age
print("Estimated Coefficients:")
print("b0 = ", b0, "\nb1 = ", b1)

plt.scatter(age, income, color="b", marker="o")
response_Vec = b0 + b1 * age
plt.plot(age, response_Vec, color="r")
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()
