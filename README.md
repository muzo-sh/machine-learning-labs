
# Machine Learning & Data Analysis Coursework Labs 🧪

A structured collection of Python practical lab exercises and algorithmic implementations completed during university coursework.

### 1. `01-linear-regression/`
- **Files:** `linear regression.py`, `linear regression2.py`, `LifeExpectancy.csv`
- **Description:** Implements single-variable and multi-variable linear regression models using `scikit-learn`. Analyzes global health indicators (Schooling, Income composition, BMI, Adult Mortality, GDP) to estimate and evaluate life expectancy trends, quantifying out-of-sample prediction errors and variance.

### 2. `02-decision-trees/`
- **File:** `decisiontree.py`
- **Description:** Trains a `DecisionTreeClassifier` on the standard Iris dataset. Evaluates model stability by testing 10 different random states, experimenting with 5 distinct train/test split ratios, and programmatically plotting accuracy curves and the final decision tree architecture.

### 3. `03-kmeans-clustering/`
- **Files:** `cluster.py`, `s1.txt` through `s4.txt`, `spiral.txt`
- **Description:** Applies unsupervised **K-Means Clustering** to standard 2D coordinate distributions (`s1`–`s4` with 15 clusters) and challenges non-linear spatial boundaries using a complex 3-cluster spiral dataset (`spiral.txt`), comparing predicted guesses against ground-truth labels.

### 4. `04-naive-bayes-mnist/`
- **File:** `naiveprc.py`
- **Description:** Downloads the MNIST handwritten digits dataset (`mnist_784`) via OpenML, logs dataset structures to file using Python's `logging` module, trains a **Multinomial Naive Bayes** classifier, and evaluates per-class error rates via confusion matrices and bar charts.

### 5. `05-neural-networks-fashion-mnist/`
- **Files:** `Neural Network.py`, `fashion_cnn_model.keras`
- **Description:** Builds, trains, and compares a **Fully Connected Network (FCN)** versus a **Convolutional Neural Network (CNN)** on the Fashion-MNIST image dataset. Evaluates architecture trade-offs, visualizes epoch accuracy/loss curves, analyzes confusion matrices for visually similar apparel classes, and demonstrates model serialization and reloading.

## ⚙️ Requirements
To run any of the labs, make sure you have the standard data science environment installed:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
