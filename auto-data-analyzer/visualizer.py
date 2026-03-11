import streamlit as st
import plotly.express as px

def auto_visualize(df):

    numeric = df.select_dtypes(include="number").columns

    cat = df.select_dtypes(include="object").columns

    for col in numeric:

        fig = px.histogram(df,x=col,title=f"{col} Distribution")

        st.plotly_chart(fig,use_container_width=True)

    for col in numeric:

        fig = px.box(df,y=col,title=f"{col} Box Plot")

        st.plotly_chart(fig,use_container_width=True)

    if len(numeric)>=2:

        fig = px.scatter(df,x=numeric[0],y=numeric[1],
        title="3D Scatter Visualization")

        st.plotly_chart(fig,use_container_width=True)

    if len(numeric)>=3:

        fig = px.scatter_3d(
            df,
            x=numeric[0],
            y=numeric[1],
            z=numeric[2]
        )

        st.plotly_chart(fig,use_container_width=True)