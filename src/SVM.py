import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Feature Engineering: Creating Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_Scaled)
X_test_poly = poly.transform(X_test_Scaled)

# Addressing Class Imbalance with SMOTE
smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train_poly, y_train)

# Hyperparameter Tuning with RandomizedSearchCV
param_distributions = {
    'C': expon(scale=100),
    'gamma': expon(scale=0.1),
    'kernel': ['rbf', 'poly', 'sigmoid']
}
random_search = RandomizedSearchCV(SVC(), param_distributions, n_iter=5, refit=True, verbose=2, cv=2, random_state=0, n_jobs=-1)
random_search.fit(X_train_smote, y_train_smote)

# Best model parameters
print("Best parameters found:", random_search.best_params_)
best_model = random_search.best_estimator_

# Evaluation
predictions = best_model.predict(X_test_poly)
Confusion_Matrix = confusion_matrix(y_test, predictions)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
plot_cm.plot()
plt.show()

# ROC Curve
decision_scores = best_model.decision_function(X_test_poly)
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
    'Model': ['SVM with Feature Engineering and SMOTE'],
    'Accuracy': [accuracy_score(y_test, predictions)],
    'Precision': [precision_score(y_test, predictions)],
    'Recall': [recall_score(y_test, predictions)],
    'F1 score': [f1_score(y_test, predictions)],
    'AUC': [auc_score]
}

results_df = pd.DataFrame(results)
print(results_df)

# Saving results
results_df.to_csv("data/svm_with_fe_smote.csv", index=False)
