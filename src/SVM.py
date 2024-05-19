import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from scipy.stats import expon
from imblearn.over_sampling import SMOTE

# Loading and sampling the data
df = pd.read_csv('data/preprocessed_data.csv')
df_sampled = df.sample(frac=0.3, random_state=1)

X = df_sampled.drop('income', axis=1)
y = df_sampled['income']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.transform(X_test)


# Hyperparameter Tuning with RandomizedSearchCV
param_distributions = {
    'C': [0.1,1,10,20],
    'gamma': ['auto', 'scale'],
    'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
}
grid_search = GridSearchCV(SVC(),  param_grid=param_distributions, cv=5, n_jobs=-1, verbose=10, scoring='f1')
grid_search.fit(X_train_Scaled, y_train)

# Best model parameters
print("Best parameters found:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluation
predictions = best_model.predict(X_test_Scaled)
Confusion_Matrix = confusion_matrix(y_test, predictions)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
plot_cm.plot()
plt.show()

# ROC Curve
decision_scores = best_model.decision_function(X_test_Scaled)
specificity, sensitivity, thresholds = roc_curve(y_test, decision_scores)
auc_score = roc_auc_score(y_test, decision_scores)
plt.figure()
plt.plot(specificity, sensitivity, label='ROC curve (area = %.2f)' % auc_score)
plt.plot([0, 1], [0, 1], '--')
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
    'Model': ['SVM'],
    'Accuracy': [accuracy_score(y_test, predictions)],
    'Precision': [precision_score(y_test, predictions)],
    'Recall': [recall_score(y_test, predictions)],
    'F1 score': [f1_score(y_test, predictions)],
    'AUC': [auc_score]
}

results_df = pd.DataFrame(results)
print(results_df)

# Saving results
results_df.to_csv("data/svm.csv", index=False)