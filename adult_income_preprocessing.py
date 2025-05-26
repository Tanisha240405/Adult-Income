import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
df = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)

# 1. Explore the dataset
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# 2. Handle missing values (drop rows with missing data for simplicity)
df.dropna(inplace=True)
print("\nShape after dropping missing rows:", df.shape)

# 3. Convert categorical features to numeric
# Label encode binary and one-hot encode multi-class
binary_cols = ['sex', 'income']
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# One-hot encode multi-class categoricals
df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation',
                                 'relationship', 'race', 'native-country'], drop_first=True)

# 4. Normalize/standardize numerical columns
scaler = StandardScaler()
numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 5. Visualize and remove outliers using boxplots
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Remove outliers using IQR method
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Final results
print("\nCleaned Data Shape:", df.shape)
print("Final Columns:\n", df.columns.tolist())
print("\nSample Data:\n", df.head())

df.to_csv("adultincome_cleaned.csv",index=False)
print("\nCleaned data saved to 'adultincome_cleaned.csv'")