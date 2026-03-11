import pandas as pd

def clean_data(df):

    df = df.drop_duplicates()

    for col in df.columns:

        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())

    return df


def analyze_data(df):

    analysis = {}

    analysis["shape"] = df.shape
    analysis["columns"] = list(df.columns)
    analysis["missing"] = df.isnull().sum()
    analysis["summary"] = df.describe()

    return analysis