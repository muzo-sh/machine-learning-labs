import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


print("Starting Part 2: Multiple Linear Regression...\n")

# 1. Load the data
file_name = 'LifeExpectancy.csv'
df = pd.read_csv(file_name)

# 2. We choose our 4 best parameters + the target (Fixed the spaces in ' BMI ')
cols_needed = ['Life expectancy ', 'Schooling', 'Income composition of resources', ' BMI ', 'Adult Mortality']
df_clean = df.dropna(subset=cols_needed)

# 3. Split the data
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)

# Define X (our 4 parameters) and y (target) for Train set
X_train = train_df[['Schooling', 'Income composition of resources', ' BMI ', 'Adult Mortality']]
y_train = train_df[['Life expectancy ']]

# Define X (our 4 parameters) and y (target) for Test set
X_test = test_df[['Schooling', 'Income composition of resources', ' BMI ', 'Adult Mortality']]
y_test = test_df[['Life expectancy ']]

# Fit model on TRAIN data separately

print("================ MODEL 1: FITTED ON TRAIN DATA ================")
model_train = linear_model.LinearRegression()
model_train.fit(X_train, y_train)

print("Coefficients (Slopes for the 4 parameters):", model_train.coef_[0])
print(f"Intercept: {model_train.intercept_[0]:.5f}")
print(f"Score (R-squared) on Train data: {model_train.score(X_train, y_train):.5f}")

# Predict the test set using the Train model and calculate errors
y_pred_from_train = model_train.predict(X_test)
errors_train = np.abs(y_test.values - y_pred_from_train)

print("\n--- Errors when using Train Model to predict Test Data ---")
print(f"Average Error: {np.mean(errors_train):.4f} years")
print(f"Standard Deviation of Errors: {np.std(errors_train):.4f} years\n")

# Fit model on TEST data separately
print("================ MODEL 2: FITTED ON TEST DATA ================")
model_test = linear_model.LinearRegression()
model_test.fit(X_test, y_test)

print("Coefficients (Slopes for the 4 parameters):", model_test.coef_[0])
print(f"Intercept: {model_test.intercept_[0]:.5f}")
print(f"Score (R-squared) on Test data: {model_test.score(X_test, y_test):.5f}")

# Predict the test set using the Test model (checking how it does on its own data)
y_pred_from_test = model_test.predict(X_test)
errors_test = np.abs(y_test.values - y_pred_from_test)

print("\n--- Errors when using Test Model to predict Test Data ---")
print(f"Average Error: {np.mean(errors_test):.4f} years")
print(f"Standard Deviation of Errors: {np.std(errors_test):.4f} years")

plt.figure(figsize=(7, 4))
plt.scatter(y_test, y_pred_from_train, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.title('Multiple Linear Regression: Actual vs Predicted Life Expectancy')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.legend()
plt.tight_layout()
plt.show()