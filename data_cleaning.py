import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Loading Dataset
print("Loading dataset...")
df = pd.read_csv('data.csv')

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

# Data Cleaning
print("\n" + "="*50)
print("STARTING DATA CLEANING")
print("="*50)

# Convert date columns
date_columns = ['Order Date', 'Ship Date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"Converted {col} to datetime")

# Handle missing values
print(f"\nMissing values before cleaning: {df.isnull().sum().sum()}")

# Remove rows with missing critical information
critical_cols = ['Customer ID', 'Sales', 'Profit']
df_clean = df.dropna(subset=critical_cols)
print(f"Rows after removing missing critical data: {len(df_clean)}")

# Remove duplicates
df_clean = df_clean.drop_duplicates()
print(f"Rows after removing duplicates: {len(df_clean)}")

# Clean numeric columns
numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
for col in numeric_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Remove outliers (using IQR method)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from Sales and Profit
original_len = len(df_clean)
df_clean = remove_outliers(df_clean, 'Sales')
df_clean = remove_outliers(df_clean, 'Profit')
print(f"Rows after removing outliers: {len(df_clean)} (removed {original_len - len(df_clean)} outliers)")

# Create additional features for analysis
df_clean['Order Year'] = df_clean['Order Date'].dt.year
df_clean['Order Month'] = df_clean['Order Date'].dt.month
df_clean['Profit Margin'] = (df_clean['Profit'] / df_clean['Sales']) * 100

print("\nData cleaning completed!")
print(f"Final dataset shape: {df_clean.shape}")

# Save cleaned data
df_clean.to_csv('cleaned_data.csv', index=False)
print("\nCleaned data saved as 'cleaned_data.csv'")

# Display summary of cleaned data
print("\n" + "="*50)
print("CLEANED DATA SUMMARY")
print("="*50)
print(f"Total transactions: {len(df_clean):,}")
print(f"Unique customers: {df_clean['Customer ID'].nunique():,}")
print(f"Date range: {df_clean['Order Date'].min()} to {df_clean['Order Date'].max()}")
print(f"Total sales: ${df_clean['Sales'].sum():,.2f}")
print(f"Total profit: ${df_clean['Profit'].sum():,.2f}")
print(f"Average profit margin: {df_clean['Profit Margin'].mean():.2f}%")