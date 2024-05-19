import keras.metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras import optimizers 
from keras.optimizers import Adam
from kerastuner.tuners import RandomSearch, GridSearch, Hyperband
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_score, recall_score, f1_score


df = pd.read_csv('data/preprocessed_data.csv')

X= df.drop('income', axis=1)
y= df['income']

#splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state= 0)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=8, max_value=64, step=8),
                    activation='relu', input_shape=(X_train_scaled.shape[1],)))

    # Adding hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'units_hidden_{i}', min_value=8, max_value=64, step=8),
                        activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(
                      hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  metrics=['accuracy', keras.metrics.AUC])
    return model


tuner = RandomSearch(
    build_model,
    objective=['val_auc','val_accuracy'],
    max_trials=5,
    directory='keras_tuner_trials',
    project_name='your_model')

tuner.search(X_train_scaled, y_train,
             epochs=50,
             validation_data=(X_test_scaled, y_test))

best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hyperparameters.values)

# best_model.fit(X_train_scaled, y_train, epochs = 50,batch_size = 10, validation_data=(X_test_scaled, y_test))

# Evaluating the best model on the test set
score = best_model.evaluate(X_test_scaled, y_test, verbose=1)

print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')
print(f'Test AUC: {score[2]}')



# Predicting probabilities for the test data.
predictions_probabilities = best_model.predict(X_test)

# Converting probabilities to binary predictions
binary_predictions = [1 if p > 0.5 else 0 for p in predictions_probabilities]

# Confusion matrix
Confusion_Matrix = confusion_matrix(y_test, binary_predictions)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix)
plot_cm.plot()
plt.show()



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