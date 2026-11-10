import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. Load Fashion-MNIST dataset
# ==========================================
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Class names for reference
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ==========================================
# 2. Check how the dataset looks
# ==========================================
print("--- Dataset Properties ---")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Image resolution: {X_train.shape[1]}x{X_train.shape[2]}")
print(f"Number of classes: {len(np.unique(y_train))}")
print(f"Datatype: {X_train.dtype}")
print(f"Pixel value range: [{X_train.min()}, {X_train.max()}]")
print("--------------------------\n")

# ==========================================
# 3. Display example images
# ==========================================
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Short reflection on distinguishing classes:
# Some classes like Pullover, Coat, and Shirt look very similar visually and
# are expected to be difficult for a basic model to tell apart.

# ==========================================
# 4. Normalize input data (Min-Max Scaling)
# ==========================================
X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0

# ==========================================
# 5. Prepare data for Fully Connected Network
# ==========================================
# Instead of manual flattening, we will use a layers.Flatten() layer in Keras.
# SparseCategoricalCrossentropy allows us to keep integer labels as they are.

# ==========================================
# 6. Create Fully Connected Network (FCN)
# ==========================================
fcn_model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # Flatten images into vectors
    layers.Dense(128, activation='relu'),   # Hidden layer
    layers.Dense(64, activation='relu'),    # Additional hidden layer
    layers.Dense(10, activation='softmax')  # Output layer with softmax
])

fcn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 7. Train the model
# ==========================================
# validation_split extracts validation data automatically
history_fcn = fcn_model.fit(
    X_train_norm, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2
)

# ==========================================
# 8 & 9. Show metrics and plot accuracy/loss
# ==========================================
# Final epoch values
print("\n--- FCN Final Training Metrics ---")
print(f"Training Accuracy:   {history_fcn.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {history_fcn.history['val_accuracy'][-1]:.4f}")
print(f"Training Loss:       {history_fcn.history['loss'][-1]:.4f}")
print(f"Validation Loss:     {history_fcn.history['val_loss'][-1]:.4f}\n")

# Plotting
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_fcn.history['accuracy'], label='Train Acc')
plt.plot(history_fcn.history['val_accuracy'], label='Val Acc')
plt.title('FCN Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_fcn.history['loss'], label='Train Loss')
plt.plot(history_fcn.history['val_loss'], label='Val Loss')
plt.title('FCN Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 10 & 11. Evaluate model & Graphical Confusion Matrix
# ==========================================
fcn_loss, fcn_acc = fcn_model.evaluate(X_test_norm, y_test, verbose=0)
print(f"FCN Test Accuracy: {fcn_acc:.4f}")

# Predictions
y_pred_fcn = np.argmax(fcn_model.predict(X_test_norm), axis=-1)
cm_fcn = confusion_matrix(y_test, y_pred_fcn)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_fcn, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('FCN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

"""
--- Reflection Comment ---
- Classes classified well: 'Trouser', 'Bag', and 'Ankle boot' achieve high accuracy.
- Classes confused most often: 'Shirt' is heavily misclassified as 'T-shirt/top', 'Pullover', and 'Coat'. 
- Why mistakes appear: Visually, a shirt, a pullover, and a coat share almost identical 2D boundaries 
  and silhouettes when compressed into 28x28 grayscale resolutions, causing pixel-by-pixel structural overlap.
"""

# ==========================================
# 12. Count and display trainable parameters
# ==========================================
fcn_model.summary()

# Load and normalize
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ==========================================
# 13. Prepare data for CNN (Reshape to 2D format with channel)
# ==========================================
X_train_cnn = np.expand_dims(X_train_norm, axis=-1) # Shape becomes (60000, 28, 28, 1)
X_test_cnn = np.expand_dims(X_test_norm, axis=-1)   # Shape becomes (10000, 28, 28, 1)

# ==========================================
# 14. Create Convolutional Neural Network (CNN)
# ==========================================
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 15. Train the CNN model
# ==========================================
history_cnn = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2
)

# ==========================================
# 16 & 17. Show metrics and plot accuracy/loss
# ==========================================
print("\n--- CNN Final Training Metrics ---")
print(f"Training Accuracy:   {history_cnn.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {history_cnn.history['val_accuracy'][-1]:.4f}")
print(f"Training Loss:       {history_cnn.history['loss'][-1]:.4f}")
print(f"Validation Loss:     {history_cnn.history['val_loss'][-1]:.4f}\n")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Train Acc')
plt.plot(history_cnn.history['val_accuracy'], label='Val Acc')
plt.title('CNN Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Train Loss')
plt.plot(history_cnn.history['val_loss'], label='Val Loss')
plt.title('CNN Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 18, 19 & 20. Evaluate CNN & Confusion Matrix
# ==========================================
cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"CNN Test Accuracy: {cnn_acc:.4f}")

y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=-1)
cm_cnn = confusion_matrix(y_test, y_pred_cnn)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Oranges',
            xticklabels=class_names, yticklabels=class_names)
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Display total trainable parameters
cnn_model.summary()

# Load original data for testing
(_, _), (X_test, y_test) = fashion_mnist.load_data()
X_test_norm = X_test / 255.0
X_test_cnn = np.expand_dims(X_test_norm, axis=-1)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ==========================================
# 22. Conceptual Answers & Observations
# ==========================================
"""
--- 22. Comparison Discussion ---
- Training Time: FCN trains faster per epoch because fully connected operations are less computation-heavy 
  than 2D matrix convolutions. However, CNN converges to a better accuracy faster in terms of epoch counts.
- Number of parameters: FCN models often contain more or comparable parameter numbers due to the initial large Flatten projection 
  (784 * 128), whereas CNNs preserve parameters through spatial weight sharing in their filters.
- Confusion Matrix results: CNN performs significantly better on tough categories like 'Shirt', 'Pullover', 
  and 'Coat' because it extracts regional spatial features (edges, textures) instead of global coordinate positions.

--- Influence of Parameters ---
- Higher Number of Epochs: Increases risk of overfitting if training metrics continue improving while validation metrics stall.
- Larger Batch Size: Accelerates training updates speed, but might cause poor localization inside sharp minima.
- More Convolution Filters/Hidden Neurons: Increases model capacity, extracting richer textures but risks high overfitting.

--- Overfitting Recognition ---
- Overfitting appears when the Training Loss decreases steadily towards zero, while the Validation Loss 
  stops decreasing and starts shifting upward. It is visually recognizable on a plot when a clear gap 
  widens over continuous epochs between the training line and validation line.
"""

# ==========================================
# 23 & 25. Save and Load your Trained Model
# ==========================================
# Note: Keras models cannot be cleanly stored via traditional direct pickle because of complex underlying C++ references.
# The standard, safe way to save/load Keras models is native H5/keras serialization.
# Below we demonstrate saving using native keras methods, or pickling the weight matrices.

# Save native model structure and weights
cnn_model.save('fashion_cnn_model.keras')
print("Model saved to disk successfully.")

# Reloading back the model architecture
from tensorflow.keras.models import load_model

loaded_cnn = load_model('fashion_cnn_model.keras')
print("Model reloaded from disk successfully.")

# ==========================================
# 26. Test your loaded model on several selected images
# ==========================================
indices = [12, 25, 42, 68]  # Arbitrary array sample indices

plt.figure(figsize=(10, 3))
for idx, test_idx in enumerate(indices):
    img = X_test_cnn[test_idx]
    true_label = y_test[test_idx]

    # Expand dims to act as a batch of 1 image
    prediction = loaded_cnn.predict(np.expand_dims(img, axis=0), verbose=0)
    predicted_label = np.argmax(prediction)

    plt.subplot(1, len(indices), idx + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}")
    plt.axis('off')

plt.tight_layout()
plt.show()