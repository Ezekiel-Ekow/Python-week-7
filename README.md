# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

# Load Iris dataset
data = load_iris()
# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = pd.Categorical.from_codes(data.target, data.target_names)

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nDataset Info:")
df.info()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# No missing values detected in this dataset

# Task 2: Basic Data Analysis

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Grouping by species and calculating mean for numerical columns
grouped = df.groupby('species').mean()
print("\nGrouped Data (Mean of Numerical Columns by Species):")
print(grouped)

# Insights: Patterns in petal length/width between species are evident.

# Task 3: Data Visualization

# Line Chart (Trends in petal length for illustration purposes)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title("Petal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# Bar Chart (Average petal length per species)
plt.figure(figsize=(8, 5))
grouped['petal length (cm)'].plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram (Distribution of sepal width)
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=10, color='purple', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot (Sepal length vs Petal length)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

