import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve

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

# Train a Random Forest model for feature importance
rf_model = RandomForestClassifier()
rf_model.fit(X_train_Scaled, y_train)

# Feature importance from Random Forest
feature_importances = rf_model.feature_importances_
plt.barh(range(len(feature_importances)), feature_importances, align='center')
plt.yticks(range(len(feature_importances)), X.columns)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

# Voting classifier - simple model stacking
voting_clf = VotingClassifier(
    estimators=[('knn', knn_model), ('rf', rf_model)], voting='soft'
)
voting_clf.fit(X_train_Scaled, y_train)

# Evaluate the ensemble model
ensemble_predictions = voting_clf.predict(X_test_Scaled)
ensemble_proba = voting_clf.predict_proba(X_test_Scaled)[:, 1]

# Find the optimal threshold for F1 score
precision, recall, thresholds = precision_recall_curve(y_test, ensemble_proba)
f1_scores = 2*recall*precision / (recall+precision)
best_threshold = thresholds[np.argmax(f1_scores)]

# Apply the threshold to make predictions
optimized_predictions = (ensemble_proba >= best_threshold).astype(int)

# Classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, optimized_predictions))
Confusion_Matrix = confusion_matrix(y_test, optimized_predictions)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
plot_cm.plot()
plt.show()

# Update results with the best model's performance
results = {
    'Model': ['Voting Classifier with Optimized Threshold'],
    'Accuracy': [accuracy_score(y_test, optimized_predictions)],
    'Precision': [precision_score(y_test, optimized_predictions)],
    'Recall': [recall_score(y_test, optimized_predictions)],
    'F1 Score': [f1_score(y_test, optimized_predictions)],
    'AUC': [roc_auc_score(y_test, ensemble_proba)]
}

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("data/knn.csv", index=False)
