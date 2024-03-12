import pandas as pd
import numpy as np

#reading the dataframe
df = pd.read_csv("data\census_income.csv")

print("Information about the dataset")
print(df.info())

print("Summary of the dataset")
print(df.describe(include = "all"))

#Handling the miscallaneous value (?) in the dataset
df = df.replace('?', np.nan)

#Discrepancies present in the income value
df['income'] = df['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'}) 

#encoding <=50k as 1 and >50k as 0
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

#encoding male as 1 and female as 0
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

#checking the percentage of missing values present in each column
print("Percentage of missing values in each column")
print((df.isna().sum()/df.shape[0])*100)

#dropping the rows as there is only less percentage of rows are missing
df = df.dropna()

print("Checking the percentage of missing values in each column after dropping")
print((df.isna().sum()/df.shape[0])*100)

#analysing capital-loss and capital-gain data

print("Count of each value in capital-loss")
print(df['capital-loss'].value_counts()/df.shape[0]*100 )

print("Count of each value in capital-gain")
print(df['capital-gain'].value_counts()/df.shape[0]*100 )

#A single value is being repeated in large amounts which would not contribute in predicting the outcome
df.drop(['capital-gain','capital-loss'], axis =1, inplace = True)

#dropping reduntant or non-informative variables
df.drop(['education','fnlwgt'], axis = 1, inplace = True) 

#Only considering rows where country is united states as most of the data is US concentrated, this is to avoid imbalance data
df = df[df['native-country']=='United-States']

#Now we can drop the native-country column

df = df.drop(['native-country'], axis = 1) 

#binning the age to make the bertter analysis amongst the age groups
bins = [15,20,25,30,35, 40, 45, 50, 60,100]
df['age_binned'] = pd.cut(df['age'], bins)

#performing one hot encoding
df_encoded = pd.get_dummies(df.drop('age',axis=1), columns = ['workclass','marital-status','occupation','relationship','race','age_binned'], drop_first = True)

print("Number of duplicate rows:",df_encoded.duplicated().sum())
df_encoded= df_encoded.drop_duplicates()

#checking if there are any duplicate rows
print("Checking the unique value counts")
print(df_encoded['income'].value_counts())
print("No.of duplicate rows after dropping the income column")
print(df_encoded.drop('income', axis = 1).duplicated().sum())

#dropping the duplicates
df_encoded = df_encoded.drop_duplicates(subset=df_encoded.columns.difference(['income']), keep = False)

#Assigning False to 0 and True to 1
df_encoded = df_encoded.replace([False, True],[0,1])

#import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df_encoded.drop(['income'], axis =1 )
y = df_encoded['income']

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] =""
# calculating VIF for each feature

for i in range(len(X.columns)):
  vif_data.iloc[i,1] =variance_inflation_factor(X.values, i)
 
  
print(vif_data)

print("VIF > 10:")
print(vif_data[vif_data['VIF']>10])

X = df_encoded.drop(['income','race_White'], axis =1 )
y = df_encoded['income'] 

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
print(X.columns)
vif_data["VIF"] =""
# calculating VIF for each feature

for i in range(len(X.columns)):
  vif_data.iloc[i,1] =variance_inflation_factor(X.values, i)
print(vif_data[vif_data['VIF']>10])


X = df_encoded.drop(['income','race_White','marital-status_Married-civ-spouse'], axis =1 )
y = df_encoded['income']

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

vif_data["VIF"] =""
# calculating VIF for each feature

for i in range(len(X.columns)):
  vif_data.iloc[i,1] =variance_inflation_factor(X.values, i)
print(vif_data[vif_data['VIF']>10])

df_encoded.drop(['race_White','marital-status_Married-civ-spouse'], axis =1, inplace = True)
print(df_encoded)

df_encoded.to_csv('data\preprocesses_data.csv')