import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import plotly.express as px

def feature_importance(df,target):

    data = df.copy()

    # Convert categorical columns
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype("category").cat.codes

    X = data.drop(columns=[target])
    y = data[target]

    model = RandomForestClassifier()

    model.fit(X,y)

    importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature":X.columns,
        "Importance":importance
    }).sort_values("Importance",ascending=False)

    fig = px.bar(
        imp_df,
        x="Feature",
        y="Importance",
        title="Feature Importance"
    )

    st.plotly_chart(fig,use_container_width=True,key="feature_importance_chart")