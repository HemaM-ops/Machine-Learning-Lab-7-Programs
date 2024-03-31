import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load your dataset
dataset_path = "C://Users//Anu//Documents//SEM_04//ML//PROJECT//Legality prediction//English_Abstractive_Embeddings_Fasttext.xlsx"
data = pd.read_excel(dataset_path)

# Separate features and target variable
X = data.drop(columns=['Judgement Status'])  # Replace 'target_column' with your target column name
y = data['Judgement Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'CatBoost': CatBoostClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier(),
    'Naïve-Bayes': GaussianNB()
}

# Train and evaluate classifiers
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # For multiclass classification
    recall = recall_score(y_test, y_pred, average='macro')  # For multiclass classification
    f1 = f1_score(y_test, y_pred, average='macro')  # For multiclass classification
    if hasattr(clf, 'predict_proba'):
        roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test), average='macro', multi_class='ovr')  # For multiclass classification
    else:
        roc_auc = roc_auc_score(y_test, clf.decision_function(X_test), average='macro', multi_class='ovr')  # For multiclass classification

    results[name] = [accuracy, precision, recall, f1, roc_auc]

# Create a DataFrame to tabulate the results
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC'])
print(results_df)
