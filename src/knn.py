import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve

# Load and prepare the data
df = pd.read_csv('data/preprocessed_data.csv')
X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.transform(X_test)

# Train a KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_Scaled, y_train)

# Adjusting hyperparameter grid to include class_weight
param_grid = {
    'n_neighbors': [x for x in range(1,10)],
    'weights': ['uniform','distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1,2]
}

# Grid search CV focusing on F1 score
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=10, scoring='f1')
grid_search.fit(X_train_Scaled, y_train)

# Getting the best possible model
best_model = grid_search.best_estimator_
print("Best Estimators:")
print(best_model)

# Performing predictions with the optimized model
predictions = best_model.predict(X_test_Scaled)

# Classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, predictions))
Confusion_Matrix = confusion_matrix(y_test, predictions)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
plot_cm.plot()
plt.show()

# Update results with the best model's performance
results = {
    'Model': ['KNN'],
    'Accuracy': [accuracy_score(y_test, predictions)],
    'Precision': [precision_score(y_test, predictions)],
    'Recall': [recall_score(y_test, predictions)],
    'F1 Score': [f1_score(y_test, predictions)],
    'AUC': [roc_auc_score(y_test, predictions)]
}

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("data/knn.csv", index=False)