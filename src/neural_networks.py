import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras import optimizers 
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score


df = pd.read_csv('data\preprocessed_data.csv')

X= df.drop('income', axis=1)
y= df['income']

#splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state= 0)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the constructor
model = Sequential()

# Adding an input layer
model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],)))

# Adding one hidden layer
model.add(Dense(8, activation='relu'))

# Adding an output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

# Evaluating the model on the test set
score = model.evaluate(X_test, y_test, verbose=1)

print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')


# Making predictions
predictions = model.predict(X_test)

# Converting probabilities to binary predictions
binary_predictions = [1 if p > 0.5 else 0 for p in predictions.flatten()]

# Confusion matrix
Confusion_Matrix = confusion_matrix(y_test, binary_predictions)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
plot_cm.plot()
plt.show()


# Predicting probabilities for the test data.
predictions_probabilities = model.predict(X_test).flatten()

# Computing the ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_test, predictions_probabilities)
roc_auc = roc_auc_score(y_test, predictions_probabilities)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# Classification Report
print("Classification Report:\n", classification_report(y_test, binary_predictions))

# Metrics
results = {
    'Model': ['Neural Network'],
    'Accuracy': [accuracy_score(y_test, binary_predictions)],
    'Precision': [precision_score(y_test, binary_predictions)],
    'Recall': [recall_score(y_test, binary_predictions)],
    'F1 Score': [f1_score(y_test, binary_predictions)],
    'AUC': [roc_auc]
}

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("data\\neural.csv", index = False) 