import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ____________________ BAR CHART OF CATETGORICAL VARIABLE

def visualize_categorical_distribution(df, column_name, figsize = (10,4) ):
    fig = plt.figure(figsize=figsize)
    category_counts = df[column_name].value_counts()
    category_counts.plot(kind='bar', color='skyblue')
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    return fig

# ____________________ HIST OF NUMERICAL VARIABLE

def visualize_numeric_distribution_histogram(df, column, figsize = (10,4) ):
    fig = plt.figure(figsize=figsize)
    df[column].hist(bins=20, edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    return fig

# ____________________ PAIR PLOT OF NUMERICAL VARIABLE  

def visualize_numeric_relationship_pairplot(df, numeric_cols, hue, figsize = (4,4) ):
    fig = plt.figure(figsize=figsize)
    sns.pairplot(data=df[numeric_cols], hue = hue)
    return fig


# ____________________ HEAT MAP OF NUMERICAL VARIABLE  
    
def visualize_numeric_relationship_heatmap(df, numeric_cols, figsize = (6,4) ):
    correlation_matrix = df[numeric_cols].corr()
    fig = plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    return fig
