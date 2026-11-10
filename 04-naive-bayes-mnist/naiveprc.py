import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --- 1. Logging Setup ---
logging.basicConfig(filename='mnist_info.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# --- 2. Load Dataset ---
print("Downloading MNIST dataset (this may take a minute)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target

num_samples, num_features = X.shape
info_message = f"Dataset downloaded. Total samples: {num_samples}, Features per sample: {num_features}."
print(info_message)

logging.info("--- MNIST Dataset Info ---")
logging.info(info_message)
logging.info(f"X (Data) shape: {X.shape}")
logging.info(f"y (Target) shape: {y.shape}")

# --- 3. Plot Initial Samples ---
print("\nPlotting sample images...")
fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for i in range(5):
    image_matrix = X[i].reshape(28, 28)
    axes[i].imshow(image_matrix, cmap='gray')
    axes[i].set_title(f"Label: {y[i]}", fontsize=14, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# --- 4. Train-Test Split & Model Training ---
print("\n--- Splitting Data & Training Model ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("Training Multinomial Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# --- 5. Analysis & Confusion Matrix ---
print("\n--- Results & Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_model.classes_)
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("MNIST - Multinomial Naive Bayes Confusion Matrix")
plt.show()

# --- 6. Error Rates per Class ---
print("\n--- Error Rates per Class ---")
correct_predictions = cm.diagonal()
total_samples = cm.sum(axis=1)
error_rates = 1 - (correct_predictions / total_samples)

for digit, error in zip(nb_model.classes_, error_rates):
    print(f"Digit {digit}: Error Rate = {error * 100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# --- 7. Error Rates Bar Chart ---
print("\nPlotting error rates bar chart...")
classes = nb_model.classes_
error_percentages = error_rates * 100

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(classes, error_percentages, color='coral', edgecolor='black')

ax.set_xlabel('Digit Classes (0-9)', fontsize=12, fontweight='bold')
ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('MNIST Naive Bayes: Error Rate per Digit', fontsize=14, fontweight='bold')
ax.set_xticks(classes)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.2,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# --- 8. Specific Digit Filtering (Example: 2) ---
print("\nSearching and plotting samples for digit '2'...")
target_digit = '2'
indices = np.where(y == target_digit)[0]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for i, true_index in enumerate(indices[:5]):
    image_matrix = X[true_index].reshape(28, 28)
    axes[i].imshow(image_matrix, cmap='gray')
    axes[i].set_title(f"Index: {true_index}\nLabel: {y[true_index]}", fontsize=12, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.show()