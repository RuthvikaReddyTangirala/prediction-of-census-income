import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

# Reading the data
df = pd.read_csv('data/preprocessed_data.csv')

# Predictors
X = df.drop('income', axis=1)
# Target
y = df['income']

# Splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling for standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Imbalance class handling using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Reduced logistic regression hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced'],
    'max_iter': [2000]
}

# Performing grid search CV with all the available processors
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

# Getting the best possible model
best_model = grid_search.best_estimator_

# Predicting probabilities to plot ROC Curve
y_scores = best_model.predict_proba(X_test_scaled)[:, 1]

# Determining the best threshold for F1 score
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores = 2 * recall * precision / (recall + precision)
best_threshold = thresholds[np.argmax(f1_scores)]

# Predicting using the best threshold
y_pred_optimal = (y_scores >= best_threshold).astype(int)

# Model Evaluation
precision = precision_score(y_test, y_pred_optimal)
recall = recall_score(y_test, y_pred_optimal)
f1 = f1_score(y_test, y_pred_optimal)
auc = roc_auc_score(y_test, y_scores)

# Confusion matrix
Confusion_Matrix = confusion_matrix(y_test, y_pred_optimal)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
plot_cm.plot()
plt.show()

# Plotting ROC Curve
specificity, sensitivity, thresholds = roc_curve(y_test, y_scores)
plt.figure()
plt.plot(specificity, sensitivity, label=f'ROC curve (area = {auc:.2f})')

# Random predictions curve i.e., when probability is 0.5
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred_optimal))

# Metrics
results = {
    'Model': ['Logistic Regression'],
    'Accuracy': [accuracy_score(y_test, y_pred_optimal)],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'AUC': [auc]
}

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("data/lr.csv", index=False)
