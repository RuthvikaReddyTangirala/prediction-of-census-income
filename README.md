# prediction-of-census-income

### Overview

This project aims to predict the income level of individuals based on demographic data using various machine learning models. The goal is to classify individuals into two categories: those earning more than $50,000 annually and those earning less than or equal to $50,000.

### Installation

To get started with this project, you'll need to have Python installed. Clone the repository and install the required packages using 'pip':

git clone https://github.com/RuthvikaReddyTangirala/prediction-of-census-income.git
cd prediction-of-census-income
pip install -r requirements.txt

### Dataset

The dataset used in this project is the UCI Machine Learning Repository's "Adult" dataset, also known as the "Census Income" dataset. It contains 48,842 instances and 14 attributes. The attributes include age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, native country, and the income label.

#### Models

Implemented and compared several machine learning models in this project:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Neural Network
- KNN

The performance of these models is evaluated based on accuracy, precision, recall, and F1-score.

### Results

The SVM and Logistic Regression models perform best across most metrics, with SVM having a slight edge in precision, and Logistic Regression leading in recall, F1Score and tying with SVM for the highest AUC at 0.89, indicating strong predictive capabilities.

