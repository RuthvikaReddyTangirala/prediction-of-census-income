import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

df = pd.read_csv('data\preprocessed_data.csv')

X= df.drop('income', axis=1)
y= df['income']
#splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state= 0)

#Feature scaling for standardisation
scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.transform(X_test) 

#Initialising the model
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_Scaled, y_train)

#Performing predictions on the test set
predictions = model.predict(X_test_Scaled) 

#Confusion Matrix
Confusion_Matrix = confusion_matrix(y_test, predictions)
plot_cm = ConfusionMatrixDisplay(confusion_matrix = Confusion_Matrix)
plot_cm.plot()

#Plotting ROC Curve
probability_scores = model.predict_proba(X_test_Scaled)[:, 1]
specificity, sensitivity, thresholds = roc_curve(y_test, probability_scores)
auc_score = roc_auc_score(y_test, probability_scores)
plt.figure()
plt.plot(specificity, sensitivity, label='ROC curve (area = %0.2f)' % auc_score)
# random predictions curve i.e., when probability is 0.5
plt.plot([0, 1], [0, 1], '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show() 


#Classification Report
print("Classification Report:\n", classification_report(y_test, predictions))

#Metrics
results = {
    'Model': ['Logistic Regression'],
    'Accuracy': [accuracy_score(y_test, predictions)],
    'Precision': [precision_score(y_test, predictions)],
    'Recall': [recall_score(y_test, predictions)],
    'F1 Score': [f1_score(y_test, predictions)],
    'AUC': [auc_score]
}

results_df = pd.DataFrame(results)
print(results_df)