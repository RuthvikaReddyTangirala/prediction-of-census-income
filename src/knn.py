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

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

