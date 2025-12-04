import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr

# --- Configuration ---
# 1. ERROR FIX: Changed the file path to the standard CSV extension (.csv)
FILE_PATH = 'Sales_Data.csv'
SALES_COLUMN = 'Total Sales (USD)'

# --- 1. Data Loading and Cleanup ---
print("--- 1. Data Loading and Initial Cleanup ---")

try:
    # Attempt to load the corrected file path
    df = pd.read_csv(FILE_PATH)
    print(f"Successfully loaded {FILE_PATH}. Initial shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}. Please ensure the CSV is present.")
    # Exit gracefully if the file is not found
    exit()

# Rename the date column for consistency and ensure correct data type
df = df.rename(columns={'Date': 'Order Date'})

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Handling missing values: drop rows with any missing data (simple approach for cleanup)
df.dropna(inplace=True)
print(f"Dataset shape after dropping NaNs: {df.shape}")

# Convert 'Order Date' to datetime objects
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Ensure Quantity and Price are numeric (handling potential string/object types)
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df.dropna(subset=['Quantity', 'Price'], inplace=True)

# Feature Engineering: Calculate Total Sales using NumPy for efficiency
df[SALES_COLUMN] = np.multiply(df['Quantity'].values, df['Price'].values)
print(f"Calculated new feature: '{SALES_COLUMN}'")

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- 2. Detailed Exploratory Data Analysis (EDA) ---")

# Data Information Overview
print("\nDataFrame Info (Data Types and Non-Null Counts):")
df.info()

# Descriptive Statistics for Numerical Features (using NumPy functions via Pandas)
print("\nDescriptive Statistics for Numerical Features:")
print(df[['Quantity', 'Price', SALES_COLUMN]].describe())

# Check for unique values and distribution in categorical columns
print("\nCategorical Column Analysis:")
for col in ['Product', 'Category']:
    print(f"- Number of unique {col}: {df[col].nunique()}")
    print(f"- Top 5 {col}:\n{df[col].value_counts().head()}")

# Time Series EDA: Aggregate monthly sales
df['Month_Year'] = df['Order Date'].dt.to_period('M')
monthly_sales_summary = df.groupby('Month_Year')[SALES_COLUMN].sum()
print("\nMonthly Sales Summary:")
print(monthly_sales_summary.to_frame())

# --- 3. Data Visualization ---
print("\n--- 3. Data Visualization ---")
sns.set_style("whitegrid")

# 3.1. Monthly Sales Trend
plt.figure(figsize=(12, 6))
monthly_sales_summary.plot(kind='line', marker='o', color='teal', linewidth=2)
plt.title('Monthly Sales Trend Over Time', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel(SALES_COLUMN, fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 3.2. Distribution of Product Price (Histogram)
plt.figure(figsize=(8, 5))
plt.hist(df['Price'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Product Price', fontsize=16)
plt.xlabel('Price (USD)', fontsize=12)
plt.ylabel('Frequency (Number of Transactions)', fontsize=12)
plt.tight_layout()
plt.show()

# 3.3. Sales by Category (Bar Plot)
category_sales = df.groupby('Category')[SALES_COLUMN].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=category_sales.index, y=category_sales.values, palette='viridis')
plt.title('Total Sales by Product Category', fontsize=16)
plt.xlabel('Category', fontsize=12)
plt.ylabel(SALES_COLUMN, fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 4. Correlation, Hypothesis, and Linear Regression ---
print("\n--- 4. Statistical Analysis and Linear Regression ---")

# --- Hypothesis Formulation ---
print("\n4.1. Hypothesis (H1): Higher product Price leads to a lower Quantity sold.")

# --- Correlation Analysis ---
correlation_matrix = df[['Price', 'Quantity']].corr()
price_quantity_corr = correlation_matrix.loc['Price', 'Quantity']
print(f"\n4.2. Pearson Correlation Coefficient (Price vs Quantity): {price_quantity_corr:.4f}")

# Perform Pearson r test for p-value (significance)
r_value, p_value_corr = pearsonr(df['Price'], df['Quantity'])
print(f"P-value from Pearson Test: {p_value_corr:.5f}")

# --- Basic Linear Regression Model ---
print("\n4.3. Simple Linear Regression Model: Quantity ~ Price")

# Define the dependent variable (Y) and independent variable (X)
Y = df['Quantity']
X = df['Price']

# Add a constant to the independent variable for the intercept ($\beta_0$)
X = sm.add_constant(X)

# Build and fit the OLS (Ordinary Least Squares) model
model = sm.OLS(Y, X)
results = model.fit()

# Print the full statistical summary
print("\n--- OLS Regression Results Summary ---")
print(results.summary())

# Extract key statistics for hypothesis decision
price_coeff = results.params['Price']
price_p_value = results.pvalues['Price']
r_squared = results.rsquared

# --- Hypothesis Conclusion ---
print("\n--- 4.4. Hypothesis Conclusion ---")
alpha = 0.05 # Significance level

print(f"Coefficient for Price (Beta_1): {price_coeff:.4f}")
print(f"P-value for Price Coefficient: {price_p_value:.5f}")
print(f"R-squared: {r_squared:.4f}")

if price_p_value < alpha:
    print(f"\nDecision: Reject the Null Hypothesis (H0). (P-value < {alpha})")
    if price_coeff < 0:
        print(f"Conclusion: The hypothesis (H1) that there is a significant negative relationship between Price and Quantity is ACCEPTED.")
        print(f"Interpretation: For every $1.00 increase in price, the expected quantity sold decreases by approximately {abs(price_coeff):.4f} units.")
    else:
        print(f"Conclusion: The relationship is significant, but it is a positive one (Price increase leads to Quantity increase). Hypothesis (H1) is REJECTED.")
else:
    print(f"\nDecision: Fail to Reject the Null Hypothesis (H0). (P-value >= {alpha})")
    print("Conclusion: There is no statistically significant linear relationship between Price and Quantity sold.")

# 4.5. Visualization of Regression
plt.figure(figsize=(8, 6))
sns.regplot(x='Price', y='Quantity', data=df, scatter_kws={'alpha':0.4, 'color':'gray'}, line_kws={'color':'red', 'linewidth': 2})
plt.title('Regression Plot: Quantity Sold vs. Product Price', fontsize=16)
plt.xlabel('Price (USD)', fontsize=12)
plt.ylabel('Quantity Sold', fontsize=12)
plt.show()

print("\n--- Analysis Complete ---")