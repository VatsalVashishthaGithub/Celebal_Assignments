# In this file, i have done the exploratory data analysis on the given Titanic-dataset provided on Kaggle.
# dataset name is -> "train_and_test2"


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train_and_test2")

# missing values analysis
# Checking for missing values
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# age distribution 
plt.figure(figsize=(10,6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.show()

# fare distribution 
plt.figure(figsize=(10,6))
sns.histplot(df['Fare'], bins=50, kde=True)
plt.title('Fare Distribution')
plt.show()

# Passenger Class distribution 
df['Pclass'].value_counts().plot(kind='bar')
plt.title('Passenger Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Survival Rate
df['2urvived'].value_counts(normalize=True).plot(kind='bar')
plt.title('Survival Rate')
plt.xlabel('Survived')
plt.ylabel('Percentage')
plt.xticks([0,1], ['No','Yes'], rotation=0)
plt.show()

# Survival by passeneger class
pd.crosstab(df['Pclass'], df['2urvived'], normalize='index').plot(kind='bar', stacked=True)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Class')
plt.ylabel('Proportion')
plt.legend(['Died','Survived'])
plt.show()

# Survival by Gender
pd.crosstab(df['Sex'], df['2urvived'], normalize='index').plot(kind='bar', stacked=True)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender (0=Male, 1=Female)')
plt.ylabel('Proportion')
plt.legend(['Died','Survived'])
plt.show()

# Age v/s Survival
plt.figure(figsize=(10,6))
sns.boxplot(x='2urvived', y='Age', data=df)
plt.title('Age Distribution by Survival')
plt.xlabel('Survived')
plt.xticks([0,1], ['No','Yes'])
plt.show()

# Fare v/s Survival
plt.figure(figsize=(10,6))
sns.boxplot(x='2urvived', y='Fare', data=df)
plt.ylim(0, 200) # Excluding extreme outliers for better visualization
plt.title('Fare Distribution by Survival')
plt.xlabel('Survived')
plt.xticks([0,1], ['No','Yes'])
plt.show()

# Age, Fare and Survival
plt.figure(figsize=(10,6))
sns.scatterplot(x='Age', y='Fare', hue='2urvived', data=df, alpha=0.6)
plt.title('Age vs. Fare Colored by Survival')
plt.ylim(0, 300)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
corr = df[['Age','Fare','SibSp','Parch','Pclass','Sex','2urvived']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Outlier Detection
plt.figure(figsize=(10,6))
sns.boxplot(x=df['Fare'])
plt.title('Fare Distribution with Outliers')
plt.show()

# Identifying extreme fares
df[df['Fare'] > 300][['Passengerid', 'Fare', 'Pclass', '2urvived']]

# Checking very young passengers
df[df['Age'] < 1][['Passengerid', 'Age', 'Sex', 'Pclass', '2urvived']]
