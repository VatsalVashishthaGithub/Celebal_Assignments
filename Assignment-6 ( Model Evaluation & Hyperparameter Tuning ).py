import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Example with a placeholder dataset
df = pd.read_csv('dataSet_name.csv')  
X = df.drop('target_column', axis=1)
y = df['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the models
models = {
    'RandomForest': RandomForestClassifier(),
    'SVC': SVC()
}

# Training and Evaluatin the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted')
    }

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = evaluate_model(model, X_test, y_test)

# Hyperparameter Tuning
# Example tuning RandomForest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

# Example tuning SVC
param_dist_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
rand_search_svc = RandomizedSearchCV(SVC(), param_dist_svc, cv=5, n_iter=4, random_state=42)
rand_search_svc.fit(X_train, y_train)

# Comparing the results
best_rf = grid_search_rf.best_estimator_
best_svc = rand_search_svc.best_estimator_

best_results = {
    'Tuned RandomForest': evaluate_model(best_rf, X_test, y_test),
    'Tuned SVC': evaluate_model(best_svc, X_test, y_test)
}

results.update(best_results)

# Display results
for model_name, metrics in results.items():
    print(f'\n📊 {model_name}')
    for metric, score in metrics.items():
        print(f'{metric}: {score:.4f}')
