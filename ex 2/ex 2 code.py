#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean dataset
df = pd.read_csv(r"C:\Users\HARISH\Downloads\archive\accident.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, errors='coerce')
df.dropna(inplace=True)
df = df[~df.index.duplicated(keep='first')]

# Identify survival column
survival_col = next((col for col in df.columns if 'survived' in col.lower()), None)
if not survival_col:
    raise KeyError(f"No 'Survived' column found. Available: {df.columns}")

# Compute statistics
total_cases, survived_cases = len(df), df[survival_col].sum()
overall_survival_rate = (survived_cases / total_cases) * 100
daily_survival_rate = df.groupby(df.index.date)[survival_col].mean() * 100
df["Rolling_Mean"] = df[survival_col].rolling(7, min_periods=1).mean() * 100  
survival_distribution = df[survival_col].value_counts(normalize=True) * 100

# Additional Calculations
fatal_cases = total_cases - survived_cases
fatality_rate = 100 - overall_survival_rate
median_survival_rate = daily_survival_rate.median()
max_survival_date, max_survival_rate = daily_survival_rate.idxmax(), daily_survival_rate.max()

# Print key stats
print(f"Total Cases: {total_cases} | Survivors: {survived_cases} | Fatal Cases: {fatal_cases}")
print(f"Survival Rate: {overall_survival_rate:.2f}% | Fatality Rate: {fatality_rate:.2f}%")
print(f"Median Daily Survival Rate: {median_survival_rate:.2f}%")
print(f"Highest Survival Rate: {max_survival_rate:.2f}% on {max_survival_date}")

# Combined Graph: Survival Rate with Rolling Average
plt.figure(figsize=(12, 5))
sns.lineplot(x=df.index, y=df["Rolling_Mean"], color="g", label="7-Day Rolling Avg")
sns.lineplot(x=df.index, y=df[survival_col] * 100, color="b", alpha=0.5, label="Original Data")
plt.axhline(y=median_survival_rate, color='r', linestyle='--', label="Median Survival Rate")
plt.title("Survival Rate Trend with Rolling Average")
plt.xlabel("Date")
plt.ylabel("Survival Rate (%)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[2]:


# Compute survival probability distribution
survival_distribution = df[survival_col].value_counts(normalize=True) * 100
print("\nSurvival Probability Distribution:")
print(survival_distribution) 
plt.figure(figsize=(12, 5))
sns.lineplot(x=df.index, y=df[survival_col] * 100, marker="o", color="b")
plt.title("Road Accident Survival Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[3]:


# Plot: Survival Rate Distribution
plt.figure(figsize=(8, 5))  # Set figure size
sns.barplot(x=survival_distribution.index, y=survival_distribution.values, palette="Blues")  # Create bar plot
plt.title("Survival Rate Distribution")  # Set title
plt.xlabel("Survived (0 = No, 1 = Yes)")  # Label x-axis
plt.ylabel("Percentage (%)")  # Label y-axis
plt.grid(True)  # Enable grid for better readability
plt.show()  # Display the plot


# In[4]:


# Check if 'Speed_of_Impact' column exists before plotting
if 'Speed_of_Impact' in df.columns:
    plt.figure(figsize=(10, 5))  # Set figure size
    
    # Create scatter plot for Survival Rate vs. Speed of Impact
    sns.scatterplot(x=df['Speed_of_Impact'], y=df[survival_col] * 100, alpha=0.6, color="red")
    
    plt.title("Survival Rate vs. Speed of Impact")  # Set title
    plt.xlabel("Speed of Impact (km/h)")  # Label x-axis
    plt.ylabel("Survival Rate (%)")  # Label y-axis
    plt.grid(True)  # Enable grid for better readability
    
    plt.show()  # Display the plot
else:
    print("Column 'Speed_of_Impact' not found, skipping scatter plot.")  # Print message if column is missing


# In[5]:


# Plot: Survival Rate Distribution (Histogram)
plt.figure(figsize=(12, 5))  # Set figure size

# Create histogram with KDE (Kernel Density Estimate) for better visualization
sns.histplot(df[survival_col] * 100, bins=10, kde=True, color="purple")

plt.title("Survival Rate Distribution")  # Set title
plt.xlabel("Survival Rate (%)")  # Label x-axis
plt.ylabel("Frequency")  # Label y-axis
plt.grid(True)  # Enable grid for better readability

plt.show()  # Display the plot


# In[ ]:




