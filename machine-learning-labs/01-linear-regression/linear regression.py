import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

print("Starting the program...")

# 1. Load the data
file_name = 'LifeExpectancy.csv'
df = pd.read_csv(file_name)

# 2. Keep only the columns we need for Part 1 and drop empty rows
cols_needed = ['Life expectancy ', 'GDP', 'Total expenditure', 'Alcohol']
df_clean = df.dropna(subset=cols_needed)

# 3. Split the data into Train and Test sets
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)

# 4. Print out how many records we have
print("Number of records in Train set:", len(train_df))
print("Number of records in Test set:", len(test_df))

# 5. Print statistical information about Life Expectancy
print("\n--- Life Expectancy Statistics ---")
print(df_clean['Life expectancy '].describe())

# 6. Find three countries with the highest life expectancy
print("\n--- Top 3 Countries by Life Expectancy ---")
top_3_countries = df_clean.groupby('Country')['Life expectancy '].mean().sort_values(ascending=False).head(3)
print(top_3_countries)

# 8. Define our target variable (y) from the training data
y_train = train_df[['Life expectancy ']]

# The three parameters we want to test (x)
parameters = ['GDP', 'Total expenditure', 'Alcohol']

# We will create an empty dictionary to save our trained models
saved_models = {}

print("\n================ REGRESSION MODELS ================")

# 9. Loop through each parameter, build a model, and draw a graph
for param in parameters:
    X_train = train_df[[param]]

    # Create the model and fit it to the training data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    # Save the model in our dictionary for later
    saved_models[param] = regr

    # Extract slope and intercept
    slope = regr.coef_[0][0]
    intercept = regr.intercept_[0]
    score = regr.score(X_train, y_train)

    # Print the details in the console
    print(f"\n--- Model for {param} ---")
    print(f"Slope (Coefficient): {slope:.5f}")
    print(f"Intercept: {intercept:.5f}")
    print(f"Score (R-squared): {score:.5f}")

    # 10 & 11. Create the chart and show the equation
    plt.figure(figsize=(7, 4))
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, edgecolor='k', label='Training Data')

    y_pred_line = regr.predict(X_train)
    plt.plot(X_train, y_pred_line, color='red', linewidth=3, label='Regression Line')

    sign = "+" if intercept >= 0 else "-"
    equation = f"y = {slope:.4f}x {sign} {abs(intercept):.2f}"

    plt.title(f'Life Expectancy vs {param}')
    plt.xlabel(param)
    plt.ylabel('Life Expectancy')
    plt.legend()
    plt.text(0.05, 0.9, equation, transform=plt.gca().transAxes, fontsize=14, color='red',
             bbox=dict(facecolor='white', alpha=0.9))
    plt.show()

print("\n================ TEST SET PREDICTIONS & ERRORS ================")

# 12 & 13. Predict values for the test set and calculate errors
y_test = test_df[['Life expectancy ']]

for param in parameters:
    X_test = test_df[[param]]
    regr = saved_models[param]

    y_pred = regr.predict(X_test)

    errors = np.abs(y_test.values - y_pred)

    avg_error = np.mean(errors)
    std_error = np.std(errors)

    print(f"\n--- Error Metrics for {param} Model ---")
    print(f"Average Error: {avg_error:.4f} years")
    print(f"Standard Deviation of Errors: {std_error:.4f} years")