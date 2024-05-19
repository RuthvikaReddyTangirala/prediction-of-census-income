import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

# Load the data
df = pd.read_csv('data/preprocessed_data.csv')

# Define features and target
X = df.drop('income', axis=1)
y = df['income']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for Decision Tree
param_grid = {
    'criterion': ['gini','entropy','log_loss'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Initialize the grid search
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0), param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=10)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_decision_tree_model = grid_search.best_estimator_
print("Best Estimators:")
print(best_decision_tree_model)

# Make predictions with the best model
predictions = best_decision_tree_model.predict(X_test_scaled)

# Evaluate the best model
Confusion_Matrix = confusion_matrix(y_test, predictions)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
plot_cm.plot()
plt.show()

# ROC curve
probability_scores = best_decision_tree_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probability_scores)
auc_score = roc_auc_score(y_test, probability_scores)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Classification report
print("Classification Report:\n", classification_report(y_test, predictions))

# Update results with the best model's performance
results = {
    'Model': ['Optimized Decision Tree Classifier'],
    'Accuracy': [accuracy_score(y_test, predictions)],
    'Precision': [precision_score(y_test, predictions)],
    'Recall': [recall_score(y_test, predictions)],
    'F1 Score': [f1_score(y_test, predictions)],
    'AUC': [auc_score]
}

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("data/dt.csv", index=False)