import pandas as pd

def clean_data(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns

    # Fill numeric columns with mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill categorical columns with mode
    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")

    return df


def analyze_data(df):

    analysis = {}

    analysis["shape"] = df.shape
    analysis["columns"] = list(df.columns)
    analysis["missing"] = df.isnull().sum()
    analysis["summary"] = df.describe()

    return analysis
