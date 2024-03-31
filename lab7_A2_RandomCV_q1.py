import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import uniform

# Load the dataset
dataset_path = "C://Users//Anu//Documents//SEM_04//ML//PROJECT//Legality prediction//English_Abstractive_Embeddings_Fasttext.xlsx"
data = pd.read_excel(dataset_path)

# Split the dataset into features (X) and target (y)
X = data.drop(columns=['Judgement Status'])  # Replace 'target_column' with the name of your target column
y = data['Judgement Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perceptron hyperparameters
perceptron_param_grid = {
    'alpha': uniform(0.0001, 0.01),
    'max_iter': [1000, 2000, 3000]
}

# MLP hyperparameters
mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'logistic'],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': uniform(0.001, 0.1),
    'max_iter': [1000, 2000, 3000]
}

# Perform RandomizedSearchCV for perceptron
perceptron_random_search = RandomizedSearchCV(Perceptron(), perceptron_param_grid, n_iter=10, cv=5, random_state=42)
perceptron_random_search.fit(X_train, y_train)

# Perform RandomizedSearchCV for MLP
mlp_random_search = RandomizedSearchCV(MLPClassifier(), mlp_param_grid, n_iter=10, cv=5, random_state=42)
mlp_random_search.fit(X_train, y_train)

# Print best parameters and best scores for both models
print("Perceptron - Best Parameters:", perceptron_random_search.best_params_)
print("Perceptron - Best Score:", perceptron_random_search.best_score_)
print()
print("MLP - Best Parameters:", mlp_random_search.best_params_)
print("MLP - Best Score:", mlp_random_search.best_score_)