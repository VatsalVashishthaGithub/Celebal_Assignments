import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
url = "https://gist.githubusercontent.com/VatsalVashishthaGithub/dc3a4c66b12fb6424cf1b3702add4103/raw/8f5a9ea9a1b5f6b44a3a9d9c9a7a6a3e8b5d5b5a/data.csv"
df = pd.read_csv(url)

# Set style
sns.set_theme(style="whitegrid")
# here I am going to do visualization on the data based on different categories..
# 1. Survival Rate by Gender
plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Survived', data=df, palette="Set2")
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# 2. Age Distribution by Survival
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, palette="viridis", bins=20)
plt.title("Age Distribution of Survivors vs. Non-Survivors")
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# 3. Survival Rate by Passenger Class
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=df, palette="pastel")
plt.title("Survival Count by Ticket Class")
plt.xlabel("Passenger Class (Pclass)")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# 4. Fare vs. Survival
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Fare', data=df, palette="coolwarm")
plt.title("Fare Distribution by Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare")
plt.show()
