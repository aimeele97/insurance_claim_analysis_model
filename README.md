Here's the updated README with a section for the prediction model added. 
# Vehicle Insurance Claim Analysis - Prediction

This project analyzes a dataset related to vehicle insurance claims to derive insights and potentially inform policy adjustments.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Data Processing](#data-processing)
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Prediction Model](#prediction-model)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

The purpose of this analysis is to understand various factors affecting vehicle insurance claims, leveraging a dataset containing detailed information about vehicle policies, the policyholders, and the vehicles themselves.

## Dataset Overview

The dataset (`train.csv`) contains **58,592** entries and **44** features, providing a comprehensive overview of various attributes related to vehicle insurance policies.

### Sample Features

- `policy_id`: Unique identifier for the policy
- `policy_tenure`: Duration of the policy in years
- `age_of_car`: Age of the car in years
- `age_of_policyholder`: Age of the policyholder in years
- `area_cluster`: Cluster of the area where the policyholder resides
- `is_claim`: Indicator of whether a claim was made (1 for yes, 0 for no)

## Data Processing

### Initial Checks

The dataset is first loaded, and basic information such as shape and data types is displayed. Any missing values or duplicates are checked to ensure data integrity.

```python
df = pd.read_csv('train.csv')
print(df.shape)
print(df.info())
```

### Summary Statistics

A descriptive analysis of the dataset is conducted to understand the distributions of numeric features.

```python
df.describe().T
```

## Features

The dataset contains the following types of features:

- **Categorical Features**: `area_cluster`, `segment`, `fuel_type`, etc.
- **Numerical Features**: `policy_tenure`, `age_of_car`, `population_density`, etc.
- **Binary Features**: `is_claim`, `is_esc`, `is_adjustable_steering`, etc.

### Unique Value Counts

The number of unique values for each feature is also calculated to assess feature richness.

```python
df.nunique()
```

## Exploratory Data Analysis (EDA)

Preliminary insights are derived through various groupings and comparisons, particularly focusing on the relationship between the `segment` of the vehicle and its `displacement` and claim rate.

Distribution of claims

![alt text](<img/Screenshot 2024-10-22 at 9.46.54 AM.png>)

Correlation matrix

![alt text](<img/Screenshot 2024-10-20 at 3.21.23 PM.png>)

Example of the hist plot for numeric features

![alt text](<img/Screenshot 2024-10-22 at 9.43.50 AM.png>)
Example count plot for categories features

![alt text](<img/Screenshot 2024-10-20 at 3.23.47 PM.png>)
## Prediction Model

### Model Development

A machine learning model is built to predict the likelihood of a claim being made based on the features in the dataset. The following steps are followed:

1. **Data Preprocessing**: 
   - Encode categorical variables
   - Scale numerical features

2. **Train-Test Split**: Split the dataset into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=['is_claim'])
y = df_encoded['is_claim']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

3. **Model Development**

Import SMOTE from imbalanced-learn
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to your data (X = features, y = target variable)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

4. **Model Training**: Train the model on the training dataset.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

5. **Model Evaluation**: Evaluate the model using accuracy, confusion matrix, and classification report.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
![alt text](<img/Screenshot 2024-10-20 at 3.17.34 PM.png>)

Plot the actual and predicted claim
![alt text](<img/Screenshot 2024-10-20 at 3.17.42 PM.png>)
Feature importances

![alt text](<img/Screenshot 2024-10-20 at 3.17.54 PM.png>)

### Results

The results will be analyzed to determine the effectiveness of the model and how well it can predict vehicle insurance claims.

## Conclusion

This analysis aims to provide insights that can help improve the vehicle insurance claims process, potentially leading to better pricing strategies and risk assessments.

## License

MIT license

Copyright (c) [2024] [Aimee Le]
