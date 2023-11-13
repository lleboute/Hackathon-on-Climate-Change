import pandas as pd
import numpy as np

# _*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_* CLEANING FUNCTIONS  _*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*


# ____________________ CATEGORICAL DATA

# Function to handle categorical data (one-hot encoding)
def handle_categorical_data(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)

# ____________________ DATETIME 

# Function to handle date and time data (extract features)
def handle_date_time_data(df, date_columns):
    for column in date_columns:
        df[column] = pd.to_datetime(df[column])
        df[column + '_day'] = df[column].dt.day
        df[column + '_month'] = df[column].dt.month
        df[column + '_year'] = df[column].dt.year
    return df

# ____________________ DUPLICATES

# Function to handle duplicates
def handle_duplicates(df):
    return df.drop_duplicates()

# ____________________ NULL VALUES  

# Function to handle null values
def handle_null_values(df, col):
    return df.dropna(subset=[col])

# ____________________ OUTLIERS 

# Function to handle outliers using IQR method
def handle_outliers_iqr(df, column):
    #df = df[column].dropna()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def filter_outliers_sigma(df, column_name):
    series = df[column_name].dropna()
    mean = series.mean()
    std_dev = series.std()
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    return lower_bound, upper_bound

# ____________________ REMOVE IRRELEVANT FEAUTURES

# Function to handle redundant or irrelevant features
def handle_redundant_features(df, irrelevant_columns):
    return df.drop(columns=irrelevant_columns)

# ____________________ TEXT INCONSISTENCIES

# Function to handle formatting inconsistencies
def handle_formatting_inconsistencies(df, li_text_column):
    for text_column in li_text_column :
        df[text_column] = df[text_column].str.lower()
    return df

# Function to handle text data (remove special characters)
def handle_text_data(df, text_column):
    df[text_column] = df[text_column].str.replace('[^a-zA-Z0-9\s]', '', regex=True)
    return df
# ____________________ TRANSFORMS (Scale, Log)

# Function to transform data (log transformation)
def transform_data(df, numeric_columns):
    for column in numeric_columns:
        df[column] = np.log1p(df[column])  # Applying log transformation
    return df

# Function to scale numerical features (Min-Max scaling)
def scale_numerical_features(df, numeric_columns):
    for column in numeric_columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

# _*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_* REPORTING  _*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*


# Function to generate a report for empty, incorrect format, and outliers using the 3-sigma method
def generate_format_report(df, outlier_method = 'sigma'):
    format_report = {}
    total_observations = len(df)
    numeric_cols, non_numeric_cols, categorical_cols, binary_cols, date_cols = categorize_columns(df)
    
    for column in df.columns:
        # Detect null values
        empty_count = df[column].isnull().sum()
        empty_count_percentage = "{:.2%}".format(empty_count / total_observations)

        # Detect outliers for numeric columns 
        if  (column in numeric_cols) &  (column not in date_cols) :
            if (outlier_method == 'iqr') :
                 lower_bound, upper_bound = handle_outliers_iqr(df, column)
            elif (outlier_method == 'sigma')  :
                lower_bound, upper_bound = filter_outliers_sigma(df, column)
            else : 
                lower_bound, upper_bound = -np.inf, np.inf 
        
            outliers_count = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
            outliers_percentage = "{:.2%}".format(outliers_count / total_observations)
        else :
            outliers_count = np.nan
            outliers_percentage = np.nan
            
        # Reporing
        format_report[column] = {  'empty_count': empty_count,
                                    'empty_count_percentage': empty_count_percentage,
                                    'outliers_count': outliers_count,
                                    'outliers_percentage': outliers_percentage }
    return format_report



# Function that returns lists of columns by datatypes
def categorize_columns(df):
    
    non_numeric_cols = []
    numeric_cols = []
    categorical_cols = []
    binary_cols = []
    
    for column in df.columns:
        if not df[column].isnull().all():
            if pd.api.types.is_numeric_dtype(df[column]):
                numeric_cols.append(column)
            else :
                non_numeric_cols.append(column)    
    
    for col in non_numeric_cols:
        if len(df[col].unique()) <= 2:
            binary_cols.append(col)  # Binary columns have 2 unique values
        else:
            categorical_cols.append(col)
    
    # Identify date columns (assumed to be datetime type)
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return numeric_cols, non_numeric_cols, categorical_cols, binary_cols, date_cols
