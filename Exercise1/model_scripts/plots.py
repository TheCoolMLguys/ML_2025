import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def breast_cancer_plots(df):

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




if __name__ == '__main__':
    df_breast_cancer = pd.read_csv("Exercise1/data"+ os.sep + "breast-cancer-diagnostic.shuf.lrn.csv")

    breast_cancer_plots(df_breast_cancer)

