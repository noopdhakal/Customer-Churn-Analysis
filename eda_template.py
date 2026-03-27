# ===================== EDA TEMPLATE =====================

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10,6)

# 2. Load Data
df = pd.read_csv("your_file.csv")

print("🔹 First 5 rows:")
print(df.head())

# 3. Basic Info
print("\n🔹 Shape:", df.shape)

print("\n🔹 Info:")
df.info()

print("\n🔹 Describe:")
print(df.describe())

print("\n🔹 Unique values:")
print(df.nunique())

# 4. Missing Values
missing = (df.isnull().sum() * 100 / len(df)).sort_values(ascending=False)
missing = missing[missing > 0]

print("\n🔹 Missing Values (%):")
print(missing)

if len(missing) > 0:
    missing_df = missing.reset_index()
    missing_df.columns = ['feature', 'percentage']
    
    sns.barplot(x='feature', y='percentage', data=missing_df)
    plt.xticks(rotation=90)
    plt.title("Missing Values %")
    plt.show()

# 5. Separate Columns
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=['int64','float64']).columns

print("\n🔹 Categorical Columns:", list(cat_cols))
print("🔹 Numerical Columns:", list(num_cols))

# 6. Univariate Analysis

# Categorical
for col in cat_cols:
    plt.figure()
    sns.countplot(x=col, data=df)
    plt.title(f"{col} Distribution")
    plt.xticks(rotation=45)
    plt.show()

# Numerical
for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution")
    plt.show()

# 7. Bivariate Analysis (Target-Based)
target = 'Churn'  # 🔁 change this if needed

if target in df.columns:
    
    # Categorical vs Target
    for col in cat_cols:
        plt.figure()
        sns.countplot(x=col, hue=target, data=df)
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45)
        plt.show()
    
    # Numerical vs Target
    for col in num_cols:
        plt.figure()
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f"{col} vs {target}")
        plt.show()

# 8. Correlation
if len(num_cols) > 0:
    plt.figure(figsize=(10,6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

# 9. Outlier Detection
for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"{col} Outliers")
    plt.show()

# 10. Feature Engineering (Optional Example)

# Binning example (if needed)
# df['binned_feature'] = pd.cut(df['some_column'], bins=5)

# Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

print("\n🔹 EDA Completed Successfully!")

# ===================== END =====================