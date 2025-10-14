import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Exercise0/data/personality_types_data_v2.csv')


print("=" * 60)
print("PERSONALITY DATASET - COMPREHENSIVE OVERVIEW")
print("=" * 60)

# Basic dataset information
print(f"Dataset Shape: {df.shape} (rows, columns)")
print(f"Number of Features: {(len(df.columns) - 1)}")
print(f"Variables: {df.columns.tolist()}")

print("\n" + "=" * 60)
print("DATA TYPES AND QUALITY CHECK")
print("=" * 60)

# Data types and missing values
print("\nData Types:")
print(df.dtypes.value_counts())
print(f"\nMissing Values: {df.isnull().sum().sum()}")

print("\n" + "=" * 60)
print("TARGET VARIABLE ANALYSIS - PERSONALITY TYPES")
print("=" * 60)

# Target variable analysis
print(f"Total individuals: {len(df)}")
print(f"Number of unique personality types: {df['Personality'].nunique()}")
print("\nPersonality Type Distribution:")
print(df['Personality'].value_counts())

print("\n" + "=" * 60)
print("DEMOGRAPHIC PROFILE")
print("=" * 60)

print(f"Average Age: {df['Age'].mean():.1f} years")
print(f"Age Range: {df['Age'].min()} - {df['Age'].max()} years")
print(f"Gender Distribution:\n{df['Gender'].value_counts()}")
print(f"Education Distribution (0=Undergraduate, 1=Graduate):\n{df['Education'].value_counts()}")
print(f"Interest Distribution:\n{df['Interest'].value_counts()}")

print("\n" + "=" * 60)
print("PERSONALITY SCORES SUMMARY STATISTICS")
print("=" * 60)

# Personality scores analysis
scores = ['Introversion Score', 'Sensing Score', 'Thinking Score', 'Judging Score']
print(df[scores].describe().round(2))

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

# Correlation between personality scores
correlation_matrix = df[scores].corr()
print("Correlation between personality dimensions:")
for i in range(len(scores)):
    for j in range(i+1, len(scores)):
        corr = correlation_matrix.iloc[i, j]
        print(f"  {scores[i]} vs {scores[j]}: {corr:.3f}")

print("\n" + "=" * 60)
print("KEY INSIGHTS SUMMARY")
print("=" * 60)

# Key insights
print(f"Most common personality type: {df['Personality'].value_counts().index[0]}")
print(f"Least common personality type: {df['Personality'].value_counts().index[-1]}")
print(f"Gender ratio (Male/Female): {df['Gender'].value_counts().get('Male', 0)}/{df['Gender'].value_counts().get('Female', 0)}")
print(f"Education ratio (Graduate/Undergraduate): {df['Education'].value_counts().get(1, 0)}/{df['Education'].value_counts().get(0, 0)}")

print(f"\nPersonality Dimension Analysis:")
for personality in df['Personality'].unique():
    subset = df[df['Personality'] == personality]
    print(f"  {personality}:")
    print(f"    Count: {len(subset)}")
    print(f"    Avg Introversion: {subset['Introversion Score'].mean():.1f}")
    print(f"    Avg Sensing: {subset['Sensing Score'].mean():.1f}")
    print(f"    Avg Thinking: {subset['Thinking Score'].mean():.1f}")
    print(f"    Avg Judging: {subset['Judging Score'].mean():.1f}")

print("\n" + "=" * 60)
print("VISUAL ANALYSIS")
print("=" * 60)


plt.figure(figsize=(15, 10))

# Personality type distribution (target)
plt.figure(figsize=(12, 6))
df['Personality'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Personality Type Distribution (Target Variable)')
plt.xlabel('Personality Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# OTHER FEATURES PLOTS
plt.figure(figsize=(15, 10))

# 1. Age distribution
plt.subplot(2, 3, 1)
df['Age'].hist(bins=20, color='lightgreen')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# 2. Gender distribution
plt.subplot(2, 3, 2)
df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Gender Distribution')

# 3. Personality scores distribution
plt.subplot(2, 3, 3)
scores = ['Introversion Score', 'Sensing Score', 'Thinking Score', 'Judging Score']
df[scores].boxplot()
plt.title('Personality Scores Distribution')
plt.xticks(rotation=45)

# 4. Education level
plt.subplot(2, 3, 4)
df['Education'].value_counts().plot(kind='bar', color='orange')
plt.title('Education Level (0=Undergraduate, 1=Graduate)')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=0)

# 5. Interests
plt.subplot(2, 3, 5)
df['Interest'].value_counts().plot(kind='bar', color='purple')
plt.title('Interest Distribution')
plt.xlabel('Interest')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = ['Age', 'Introversion Score', 'Sensing Score', 'Thinking Score', 'Judging Score']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Features')
plt.tight_layout()
plt.show()

# Personality vs Gender
plt.figure(figsize=(12, 6))
pd.crosstab(df['Personality'], df['Gender']).plot(kind='bar')
plt.title('Personality Types by Gender')
plt.xlabel('Personality Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("\nDataset Characteristics:")
print(f"• {len(df)} individuals with {df['Personality'].nunique()} personality types")
print(f"• Includes personality scores for MBTI dimensions")
print(f"• Contains demographic and interest information")

print("\nAnalysis Focus:")
print("• Predicting personality types from scores and demographics")
print("• Understanding relationships between MBTI dimensions")
print("• Exploring demographic patterns in personality distribution")