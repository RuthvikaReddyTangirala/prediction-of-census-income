# -*- coding: utf-8 -*-
# python3 -m pip install ucimlrepo
"""**Data Acquisition**"""

from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")


# fetch dataset
census_income = fetch_ucirepo(id=20)

# data (as pandas dataframes)
X = census_income.data.features
y = census_income.data.targets

# metadata
print(census_income.metadata)

# variable information
print(census_income.variables)

X # viewing the predictors

y # viewing the response variable

df = X.merge(y, left_index=True, right_index=True)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""**Data Inspection**"""

df.dtypes

df.info()

df.describe()

df.describe(include = 'all')

"""**Handling Missing Values**"""

df.isnull().sum()

df[df['occupation'].isnull() & (df['age'] < 14)]

df[df['workclass'] == 'Never-worked']

"""Have observed occupation column values as '?' - therefore filling them with NaN"""

# df.replace('?', np.nan, inplace=True)
# #df.loc[df['occupation'] == '?', 'occupation'] = np.nan

df[df['workclass'] == 'Never-worked']

df['income'] = df['income'].replace({'>50K.': '>50K', '<=50K.': '<=50K'})

"""Dropping all NaN value rows for now - can look into this later"""

df.dropna()

import os
import pandas as pd


# Create the directory if it doesn't exist
# os.makedirs(directory, exist_ok=True)

# Specify the file path within the directory
# file_path = os.path.join(directory, "data.xlsx")

# Use the to_excel() method to save the DataFrame as an Excel file
# df.to_excel(file_path, index=False)  # Set index=False if you don't want to include row numbers
df.to_csv("data/vis_data.csv", index=False)

print("DataFrame saved as Excel file successfully.")

"""**Dropping Duplicates**"""

df.drop_duplicates(inplace=True,keep='first')

"""**Visualizations**"""

education_counts = df['education'].value_counts()

# Extracting education levels and their corresponding frequencies
education_levels = education_counts.index
frequency = education_counts.values

# Plotting the frequencies
plt.figure(figsize=(10, 6))
plt.barh(education_levels, frequency, color='skyblue')

plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.title('Frequency of Education Levels')
plt.xticks(rotation=45)
plt.show()

# Group the data by education group and income level, then count the number of entries
grouped = df.groupby(['education', 'income']).size().unstack()

# Plot the stacked bar graph
grouped.plot(kind='barh', stacked=True, figsize=(10, 6))
plt.title('Education Group vs Number of Entries with Income > or <= 50K')
plt.ylabel('Education Group')
plt.xlabel('Number of Entries')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.show()

workclass_counts = df['workclass'].value_counts()

# Extracting workclass types and their corresponding frequencies
workclass_types = workclass_counts.index
frequency = workclass_counts.values

plt.figure(figsize=(10, 6))
plt.bar(workclass_types, frequency, color='skyblue')

plt.xlabel('Workclass Type')
plt.ylabel('Frequency')
plt.title('Frequency of Workclass Types')
plt.xticks(rotation=45)
plt.show()

# Group the data by workclass and income level, then count the number of entries
grouped = df.groupby(['workclass', 'income']).size().unstack()
grouped['total'] = grouped.sum(axis=1)

grouped = grouped.sort_values(by='total', ascending=False).drop('total', axis=1)

# Plot the stacked bar graph
grouped.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Workclass vs Number of Entries with Income > or <= 50K')
plt.xlabel('Workclass')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.show()

age_counts = df['age'].value_counts()
ages = age_counts.index
frequency = age_counts.values

plt.figure(figsize=(10, 6))
plt.scatter(ages, frequency, color='blue', marker='.')
plt.xlabel('Age')
plt.ylabel('Frequency of the Age')
plt.title('Age vs Frequency of the Age')
plt.show()

bins = [0, 18, 30, 40, 50, 60, 70, 80, 120]
labels = ['17-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']

df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

grouped = df.groupby(['age_group', 'income']).size().unstack()

grouped.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Age Group vs Number of Entries with Income > or <= 50K')
plt.xlabel('Age Group')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.show()

# Group the data by occupation and count the number of entries
grouped = df['occupation'].value_counts()

# Plot the bar graph
grouped.plot(kind='bar', figsize=(10, 6))
plt.title('Number of Entries by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)
plt.show()

grouped = df.groupby(['occupation', 'income']).size().unstack()

# Plot the stacked bar graph
grouped.plot(kind='barh', stacked=True, figsize=(10, 6))
plt.title('Occupation vs Number of Entries with Income > or <= 50K')
plt.ylabel('Occupation')
plt.xlabel('Number of Entries')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.show()

hour_counts = df['hours-per-week'].value_counts()
hours = hour_counts.index
frequency = hour_counts.values

plt.figure(figsize=(10, 6))
plt.scatter(hours, frequency, color='blue', marker='.')

plt.xlabel('Hours')
plt.ylabel('Frequency of the Hours')
plt.title('Hour vs Frequency of the Hour')
plt.show()

# Categorize 'hours-per-week' into groups
df['hours_group'] = pd.cut(df['hours-per-week'], bins=[0, 39, 40, float('inf')], labels=['<40', '=40', '>40'])

grouped = df.groupby(['hours_group', 'income']).size().unstack()

grouped.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Hours-per-week vs Number of Entries with Income > or <= 50K')
plt.xlabel('Hours-per-week Group')
plt.ylabel('Number of Entries')
plt.xticks(rotation=0)
plt.legend(title='Income')
plt.show()

grouped = df['race'].value_counts()

grouped.plot(kind='bar', figsize=(10, 6))
plt.title('Number of Entries by Race')
plt.xlabel('Race')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)
plt.show()

grouped = df.groupby(['race', 'income']).size().unstack()

grouped.plot(kind='barh', stacked=True, figsize=(10, 6))
plt.title('Race vs Number of Entries with Income > or <= 50K')
plt.ylabel('Race')
plt.xlabel('Number of Entries')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.show()

grouped = df['sex'].value_counts()

grouped.plot(kind='barh', figsize=(8, 6), color=['purple', 'lightgreen'])
plt.title('Number of Entries by Sex')
plt.ylabel('Sex')
plt.xlabel('Number of Entries')
plt.xticks(rotation=0)
plt.show()

grouped = df.groupby(['sex', 'income']).size().unstack()

grouped.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Sex vs Number of Entries with Income > or <= 50K')
plt.xlabel('Sex')
plt.ylabel('Number of Entries')
plt.xticks(rotation=0)
plt.legend(title='Income')
plt.show()

grouped = df['marital-status'].value_counts()

grouped.plot(kind='bar', figsize=(10, 6))
plt.title('Number of Entries by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)
plt.show()

grouped = df.groupby(['marital-status', 'income']).size().unstack()

grouped.plot(kind='barh', stacked=True, figsize=(10, 6))
plt.title('Marital Status vs Number of Entries with Income > or <= 50K')
plt.ylabel('Marital Status')
plt.xlabel('Number of Entries')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.show()

df['income'] = df['income'].apply(lambda x:x.replace("<=50K", "0"))
df['income'] = df['income'].apply(lambda x:x.replace(">50K", "1"))
df['income'] = df['income'].astype(int)

sns.set_theme(style = 'darkgrid', font_scale = 1.5)
sns.catplot(x = 'sex', y = 'income', data = df, kind = 'bar', col = 'race',
            height = 5, aspect = 1)
plt.show()

sns.set_theme(style = 'darkgrid', font_scale = 1)
plt.figure(figsize = (14,5))
sns.barplot(x = 'native-country', y = 'income', data = df)
plt.xticks(rotation = 90)
plt.title('Country-wise Income')
plt.show()

plt.figure(figsize = (14,5))

sns.countplot(x = 'native-country', data = df)
plt.xticks(rotation = 90)
plt.show()

plot = sns.displot(x=df['age'], color='darkblue')
plot.set_axis_labels('Age', '')
plt.tight_layout(pad=3.0)  # Adjust the padding as needed
plt.show()


plot = sns.displot(x=df['fnlwgt'], color='darkblue')
plot.set_axis_labels('fnlwgt','')
plt.tight_layout(pad=3.0)  # Adjust the padding as needed
plt.show()


plot = sns.displot(x = df['education-num'], color = 'darkblue')
plot.set_axis_labels('Education Number', '')
plt.tight_layout(pad=3.0)  # Adjust the padding as needed
plt.show()


plot = sns.displot(x = df['capital-gain'], color = 'darkblue')
plot.set_axis_labels('Capital Gain', '')
plt.tight_layout(pad=3.0)  # Adjust the padding as needed
plt.show()


plot = sns.displot(x = df['capital-loss'], color = 'darkblue')
plot.set_axis_labels('Capital Loss','')
plt.tight_layout(pad=3.0)  # Adjust the padding as needed
plt.show()


len(df[(df['education'] == 'Preschool') & (df['income'] == 0)]) # Not a single person attending only preschool have income <50k.

df['education'] = df['education'].apply(lambda x: 'School' if x == '11th' or x == '7th-8th' or x == '10th'
                                              or x == '5th-6th' or x == '9th' or x == '12th' or x == '1st-4th'
                                              or x == 'Preschool' else x)
df['education'] = df['education'].apply(lambda x: 'Associate' if x == 'Assoc-acdm' or x == 'Assoc-voc' else x)
education_map = {'School':1,
             'HS-grad':2,
             'Some-college':3,
             'Bachelors':4,
             'Prof-school':5,
             'Associate':6,
             'Masters':7,
             'Doctorate':8}
df['education'] = df['education'].map(education_map)

df

numeric_df = df.select_dtypes(include=['number'])

correlation_matrix = numeric_df.corr()
correlation_matrix

plt.figure(figsize = (10, 5))
sns.heatmap(correlation_matrix, annot = True)
plt.title('Correlation Plot')
plt.show()

for col in df.columns:
    print((col, len(df[df[col] == '?'])/len(df[col])*100))

"""The percetage of the entries with (?) is very low as compared to the length of the data of that particular columns. As we can see that the percentage in workclass, occupation, native.country are 6%, 6%, 2% respectively. Therefore, it seems that dropping the na values should be good choice but we can't be sure. So, check the score with and without dropping the columns and go for the one which gives better result."""