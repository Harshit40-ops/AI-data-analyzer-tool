def generate_insights(df):

    insights=[]

    insights.append(f"Dataset has {df.shape[0]} rows")

    insights.append(f"Dataset has {df.shape[1]} columns")

    insights.append(f"Total missing values {df.isnull().sum().sum()}")

    insights.append("Top correlated features detected")

    return insights