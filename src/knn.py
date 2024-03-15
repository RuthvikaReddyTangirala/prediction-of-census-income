import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StandardScaler
from sklearn.model_selection import KNeighborsClassifier

df = pd.read_csv('data\preprocessed_data.csv')

X= df.drop('income', axis=1)
y= df['income']
#splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state= 0)

scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.transform(X_test) 

