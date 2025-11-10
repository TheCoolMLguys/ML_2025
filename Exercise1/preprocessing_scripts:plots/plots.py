import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

##### plot for loan ######
df_loan = pd.read_csv("Exercise1/data/loan-10k.lrn.csv")

plt.figure(figsize=(10, 6))
grade_counts = df_loan['grade'].value_counts().sort_index()

bars = plt.bar(grade_counts.index, grade_counts.values, color='skyblue')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.xlabel('Loan Grade', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Loan Dataset - Distribution of Target Variable (Grade)', fontsize=14)
plt.grid(axis='y', alpha=0.3)

plt.xticks(rotation=0)
sns.despine()

plt.tight_layout()
plt.show()

print(f"Total instances: {len(df_loan)}")
print(f"Class distribution:")
for grade, count in grade_counts.items():
    percentage = (count / len(df_loan)) * 100
    print(f"Grade {grade}: {count} instances ({percentage:.1f}%)")

print(f"\nClass imbalance ratio (majority:minority): {grade_counts.max():.1f}:{grade_counts.min():.1f}")


##### plots for breast cancer ######

df = pd.read_csv("Exercise1/data"+ os.sep + "breast-cancer-diagnostic.shuf.lrn.csv")

### target variable bar chart ###
counts = df["class"].value_counts(normalize = True)
counts.plot(kind = 'bar')
    
plt.title('Breast Cancer - Distribution of Target Variable (class)')
plt.xlabel('Target')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.show()

### numerical variables correlationplot ###
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
    
sns.heatmap(corr,
            annot=False,
            cmap="coolwarm",    
            square=True,
            cbar_kws={"shrink": 0.8},
            linewidths=0.3)

plt.title("Breast Cancer - Correlation Matrix of Numerical Features", pad=20)
plt.tight_layout()
plt.show()


##### plot for ozone ######
df = pd.read_csv('Exercise1/data/ozone_level_data.csv')

fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

df['Ozone'].value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'red'])
ax1.set_title('Ozone Days vs Normal Days')
ax1.set_xlabel('0 = Normal Day, 1 = Ozone Day')
ax1.set_ylabel('Number of Days')
plt.tight_layout()
plt.show()

##### plot for personality ######

df = pd.read_csv('Exercise1/data/personality_types_data_v2.csv')

plt.figure(figsize=(12, 6))
df['Personality'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Personality Type Distribution (Target Variable)')
plt.xlabel('Personality Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()