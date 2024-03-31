import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data from xls file (replace with your file path)
import os

# Change working directory to the folder containing the Excel file
os.chdir("C:\\Users\\mahad\\OneDrive\\Desktop\\SEM 4 Courses\\ML\\Legality\\Datasets")

data = pd.read_excel("English_Extractive_Embeddings_Fasttext.xlsx")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers and performance metrics dictionary
classifiers = {
    "SVC": SVC(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

performance_metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1-Score": f1_score
}

# Evaluate classifiers and store results in a dictionary
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results[name] = {}
    for metric_name, metric in performance_metrics.items():
        results[name][metric_name] = metric(y_test, y_pred)

# Print results in a tabular format
print("Classification Results:")
print("{:20s} | {:>10s} | {:>10s} | {:>10s} | {:>10s}".format(
    "Classifier", "Accuracy", "Precision", "Recall", "F1-Score"))
print("-" * 80)
for name, metrics in results.items():
    print("{:20s} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>10.4f}".format(
        name, metrics["Accuracy"], metrics["Precision"], metrics["Recall"], metrics["F1-Score"]))
