import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data from xls file (replace with your file path)
import os

# Change working directory to the folder containing the Excel file
os.chdir("C:\\Users\\mahad\\OneDrive\\Desktop\\SEM 4 Courses\\ML\\Legality\\Datasets")

data = pd.read_excel("English_Extractive_Embeddings_Fasttext.xlsx")

# Extract features (embedded vectors) and labels
X = data.drop("Judgement", axis=1).values  # Assuming "label" column contains class labels
y = data["Judgement"].values

# Split data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# Define hyperparameter search spaces
perceptron_param_grid = {
    "eta": np.arange(0.01, 1.01, 0.1),  # Learning rate range
    "max_iter": [100, 200, 300]  # Maximum iterations
}

mlp_param_grid = {
    "solver": ["lbfgs", "adam"],  # Optimization algorithms
    "alpha": np.logspace(-4, -2, 5),  # Regularization parameter (inverse of L2 strength)
    "hidden_layer_sizes": [(100,), (100, 50)],  # Hidden layer configurations
}

# Create and run RandomizedSearchCV for Perceptron
perceptron = Perceptron(random_state=42)
perceptron_cv = RandomizedSearchCV(perceptron, perceptron_param_grid, cv=5, scoring="accuracy", n_iter=10)
perceptron_cv.fit(X_train, y_train)

# Get best hyperparameters and evaluate Perceptron
best_perceptron_params = perceptron_cv.best_params_
perceptron = Perceptron(**best_perceptron_params, random_state=42)
perceptron.fit(X_train, y_train)
perceptron_accuracy = accuracy_score(y_val, perceptron.predict(X_val))
print(f"Perceptron Accuracy (Validation): {perceptron_accuracy:.4f}")

# Create and run RandomizedSearchCV for MLP
mlp = MLPClassifier(random_state=42)
mlp_cv = RandomizedSearchCV(mlp, mlp_param_grid, cv=5, scoring="accuracy", n_iter=10)
mlp_cv.fit(X_train, y_train)

# Get best hyperparameters and evaluate MLP
best_mlp_params = mlp_cv.best_params_
mlp = MLPClassifier(**best_mlp_params, random_state=42)
mlp.fit(X_train, y_train)
mlp_accuracy = accuracy_score(y_val, mlp.predict(X_val))
print(f"MLP Accuracy (Validation): {mlp_accuracy:.4f}")

# Evaluate final performance on test set (optional)
perceptron_test_accuracy = accuracy_score(y_test, perceptron.predict(X_test))
mlp_test_accuracy = accuracy_score(y_test, mlp.predict(X_test))
print(f"Perceptron Accuracy (Test): {perceptron_test_accuracy:.4f}")
print(f"MLP Accuracy (Test): {mlp_test_accuracy:.4f}")
