import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Exercise0/data/ozone_level_data.csv')

print("=" * 50)
print("DATASET BASIC INFORMATION")
print("=" * 50)

# Basic dataset info
print(f"Dataset Shape: {df.shape} (rows, columns)")
print(f"Number of Features: {(len(df.columns)-1)})")
print(f"Date Range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

print("\n" + "=" * 50)
print("DATA TYPES AND MISSING VALUES")
print("=" * 50)

# Data types and missing values
print("\nData Types:")
print(df.dtypes.value_counts())
print(f"\nMissing Values: {df.isnull().sum().sum()}")

print("\n" + "=" * 50)
print("TARGET VARIABLE ANALYSIS")
print("=" * 50)

# Target variable analysis
print(f"Total days: {len(df)}")
print(f"Ozone days: {df['Ozone'].sum()} ({df['Ozone'].mean()*100:.1f}% of data)")
print(f"Normal days: {len(df) - df['Ozone'].sum()} ({(1-df['Ozone'].mean())*100:.1f}% of data)")

print("\n" + "=" * 50)
print("KEY FEATURES SUMMARY STATISTICS")
print("=" * 50)

# Summary statistics for key features (excluding Date column)
key_features = ['T_AV', 'WSR_AV', 'Precp', 'SLP', 'KI', 'TT']
print(df[key_features].describe().round(2))

print("\n" + "=" * 50)
print("CORRELATION WITH OZONE")
print("=" * 50)

# Top correlations with ozone (excluding Date column)
numeric_df = df.select_dtypes(include=['number'])
correlations = numeric_df.corr()['Ozone'].sort_values(ascending=False)
print("Top 5 positive correlations:")
print(correlations.head(6)[1:])
print("\nTop 5 negative correlations:")
print(correlations.tail(5))

print("\n" + "=" * 50)
print("VISUAL ANALYSIS")
print("=" * 50)


# Preprocessing

df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
print(df["Month"])
#df = df.replace("?", np.nan)

for col in df.columns.difference(["Date", "Month", "Year"]):
	df[col] = pd.to_numeric(df[col], errors="coerce")


fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

# 1. Ozone days vs normal days
df['Ozone'].value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'red'])
ax1.set_title('Ozone Days vs Normal Days')
ax1.set_xlabel('0 = Normal Day, 1 = Ozone Day')
ax1.set_ylabel('Number of Days')
plt.tight_layout()
plt.savefig("ozone_target_variable.png")

plt.clf()

fig1, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(16, 8))
# 2. Temperature comparison
df.boxplot(column='T_AV', by='Ozone', ax=ax2)
ax2.set_title('Temperature: Ozone vs Normal Days')
ax2.set_xlabel('')

# 3. Wind speed comparison
df.boxplot(column='WSR_AV', by='Ozone', ax=ax3)
ax3.set_title('Wind Speed: Ozone vs Normal Days')
ax3.set_xlabel('')

# 4. Key features correlation with ozone
correlations = df[['T_AV', 'WSR_AV', 'Precp', 'SLP', 'KI', 'Ozone']].corr()['Ozone'].drop('Ozone')
correlations.sort_values().plot(kind='barh', ax=ax4, color='green')
ax4.set_title('Feature Correlation with Ozone')
ax4.set_xlabel('Correlation with Ozone')

plt.suptitle('')
plt.tight_layout()
plt.savefig("ozone_features.png")
plt.clf()

# Simple stats comparison
print("\n" + "=" * 50)
print("KEY DIFFERENCES: OZONE DAYS VS NORMAL DAYS")
print("=" * 50)

ozone_days = df[df['Ozone'] == 1]
normal_days = df[df['Ozone'] == 0]

print(f"Average Temperature:")
print(f"  Ozone days: {ozone_days['T_AV'].mean():.1f}°C")
print(f"  Normal days: {normal_days['T_AV'].mean():.1f}°C")
print(f"  Difference: {ozone_days['T_AV'].mean() - normal_days['T_AV'].mean():.1f}°C")

print(f"\nAverage Wind Speed:")
print(f"  Ozone days: {ozone_days['WSR_AV'].mean():.1f}")
print(f"  Normal days: {normal_days['WSR_AV'].mean():.1f}")
print(f"  Difference: {ozone_days['WSR_AV'].mean() - normal_days['WSR_AV'].mean():.1f}")

print(f"\nAverage Precipitation:")
print(f"  Ozone days: {ozone_days['Precp'].mean():.2f}")
print(f"  Normal days: {normal_days['Precp'].mean():.2f}")
print(f"  Difference: {ozone_days['Precp'].mean() - normal_days['Precp'].mean():.2f}")

print(f"\nAverage Sea Level Pressure:")
print(f"  Ozone days: {ozone_days['SLP'].mean():.1f}")
print(f"  Normal days: {normal_days['SLP'].mean():.1f}")
print(f"  Difference: {ozone_days['SLP'].mean() - normal_days['SLP'].mean():.1f}")