def chat_with_data(df,q):

    if "columns" in q.lower():

        return df.columns.tolist()

    if "rows" in q.lower():

        return df.shape[0]

    if "missing" in q.lower():

        return df.isnull().sum().sum()

    return "Try asking about columns, rows, or missing values"