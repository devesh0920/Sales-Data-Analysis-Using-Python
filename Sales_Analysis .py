import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('sales_data.csv')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Add 'Total' sales column
df['Total'] = df['Quantity'] * df['Price']

# Basic Metrics
total_sales = df['Total'].sum()
top_product = df.groupby('Product')['Total'].sum().idxmax()
monthly_sales = df.resample('M', on='Date')['Total'].sum()

# Print key insights
print(f"Total Sales: ${total_sales}")
print(f"Top Selling Product: {top_product}")

# Plot Monthly Sales Trend
plt.figure(figsize=(10,6))
monthly_sales.plot()
plt.title('Monthly Sales Trend')
plt.ylabel('Sales in USD')
plt.xlabel('Month')
plt.show()

# Plot Top Products
top_products = df.groupby('Product')['Total'].sum().sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title('Top Products by Sales')
plt.xlabel('Sales in USD')
plt.show()

# Find and visualize seasonal peaks
df['Month'] = df['Date'].dt.month
monthly_category_sales = df.groupby(['Month', 'Category'])['Total'].sum().unstack()

monthly_category_sales.plot(kind='bar', stacked=True, figsize=(10,7))
plt.title('Monthly Sales by Category')
plt.ylabel('Sales in USD')
plt.xlabel('Month')
plt.show()
