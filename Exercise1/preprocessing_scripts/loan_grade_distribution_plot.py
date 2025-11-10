import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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