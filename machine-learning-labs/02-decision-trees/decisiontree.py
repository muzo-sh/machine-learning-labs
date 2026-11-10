import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

print("Starting Complete Decision Tree Practice...\n")

#part1
print("--- PART 1: Basic Training ---")
# Load the IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data (test_size=0.25 as requested)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate basic accuracy
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#part2

print("\n--- PART 2: Testing 10 Different Random States ---")
accuracies_10 = []

# Loop 10 times with random_states 0 through 9
for i in range(10):
    X_train_loop, X_test_loop, y_train_loop, y_test_loop = train_test_split(X, y, test_size=0.25, random_state=i)

    loop_model = DecisionTreeClassifier(random_state=42)
    loop_model.fit(X_train_loop, y_train_loop)

    loop_acc = accuracy_score(y_test_loop, loop_model.predict(X_test_loop))
    accuracies_10.append(loop_acc)

# Calculate the mean and standard deviation of the 10 runs
mean_acc = np.mean(accuracies_10)
std_acc = np.std(accuracies_10)

print(f"Average Accuracy over 10 runs: {mean_acc * 100:.2f}%")
print(f"Standard Deviation of Accuracy: {std_acc:.4f}")

#part3
print("\n--- PART 3: Testing Different Split Ratios ---")
ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
ratio_accuracies = []

# Loop through the specific ratios your teacher asked for
for r in ratios:
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=r, random_state=42)

    ratio_model = DecisionTreeClassifier(random_state=42)
    ratio_model.fit(X_train_r, y_train_r)

    acc = accuracy_score(y_test_r, ratio_model.predict(X_test_r))
    ratio_accuracies.append(acc)
    print(f"Accuracy with test_size={r}: {acc * 100:.2f}%")

# Draw the chart for the ratios
plt.figure(figsize=(8, 4))
plt.plot(ratios, ratio_accuracies, marker='o', color='b', linestyle='-', linewidth=2)
plt.title('Decision Tree Accuracy vs. Test Set Size Ratio')
plt.xlabel('Test Size Ratio (e.g., 0.25 = 25% test data)')
plt.ylabel('Accuracy Score')
plt.grid(True)
print("\n--> Showing Ratio Chart. CLOSE WINDOW to show the final Tree drawing!")
plt.show()

#part 4
print("\n--- PART 4: Drawing the Tree ---")
plt.figure(figsize=(12, 8))
# Use the very first model we made to draw the exemplary tree
tree.plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Exemplary Decision Tree for IRIS Dataset")
print("--> Showing Tree Drawing.")
plt.show()

print("\nDecision Tree Practice Complete!")