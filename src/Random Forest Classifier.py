import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

# Reading the data
df = pd.read_csv('data/preprocessed_data.csv')

X = df.drop('income', axis=1)
y = df['income']

# Splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature scaling for standardisation
scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.transform(X_test)

# Adjusting hyperparameter grid to include class_weight
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}, 'balanced']  # Adjusting class weight
}

# Grid search CV focusing on F1 score
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1')
grid_search.fit(X_train_Scaled, y_train)

# Getting the best possible model
best_model = grid_search.best_estimator_

# Performing predictions with the optimized model
predictions = best_model.predict(X_test_Scaled)

# Confusion Matrix
Confusion_Matrix = confusion_matrix(y_test, predictions)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
plot_cm.plot()
plt.show()

# Plotting ROC Curve
probability_scores = best_model.predict_proba(X_test_Scaled)[:, 1]
specificity, sensitivity, thresholds = roc_curve(y_test, probability_scores)
auc_score = roc_auc_score(y_test, probability_scores)
plt.figure()
plt.plot(specificity, sensitivity, label='ROC curve (area = %.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Classification Report
print('Classification Report: \n', classification_report(y_test, predictions))

# Metrics
results = {
    'Model': ['Optimized Random Forest Classifier for F1'],
    'Accuracy': [accuracy_score(y_test, predictions)],
    'Precision': [precision_score(y_test, predictions)],
    'Recall': [recall_score(y_test, predictions)],
    'F1 score': [f1_score(y_test, predictions)],
    'AUC': [auc_score]
}

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("data/RFC_optimized_f1.csv", index=False)
