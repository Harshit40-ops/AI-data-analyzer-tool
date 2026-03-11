import streamlit as st
import pandas as pd
import plotly.express as px

from analyzer import clean_data
from visualizer import auto_visualize
from model import train_models
from ml_dashboard import feature_importance
from insights import generate_insights
from chat_ai import chat_with_data
from report_generator import generate_report

st.set_page_config(layout="wide")

# ---------------- UI STYLE ----------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#000000);
background-size: 400% 400%;
animation: gradient 15s ease infinite;
}

@keyframes gradient{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

.block-container{
background: rgba(255,255,255,0.05);
padding:20px;
border-radius:15px;
backdrop-filter: blur(10px);
}

</style>
""",unsafe_allow_html=True)

# ------------------------------------------------

st.title("🚀 AI Data Analyzer ")

# ---------- SIDEBAR ----------

st.sidebar.title("Dashboard Controls")

file = st.sidebar.file_uploader("Upload Dataset")

chart_type = st.sidebar.selectbox(
"Select Chart Type",
["Scatter","Line","Bar","Histogram","3D Scatter"]
)

auto_dashboard = st.sidebar.checkbox("AI Auto Dashboard")
auto_charts = st.sidebar.checkbox("Generate 100+ Charts")

# ------------------------------------------------

if file:

    df = pd.read_csv(file)
    df = clean_data(df)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    c1,c2,c3 = st.columns(3)

    c1.metric("Rows",df.shape[0])
    c2.metric("Columns",df.shape[1])
    c3.metric("Missing",df.isnull().sum().sum())

    numeric_cols = df.select_dtypes(include="number").columns

# ---------------- AI AUTO DASHBOARD ----------------

    if auto_dashboard:

        st.subheader("🤖 AI Auto Dashboard")

        auto_visualize(df)

# ---------------- 100+ AUTO CHARTS ----------------

    if auto_charts:

        st.subheader("📊 Automatic Charts")

        for i,col in enumerate(numeric_cols):

            fig = px.histogram(df,x=col,title=f"{col} Distribution")

            st.plotly_chart(fig,use_container_width=True,key=f"hist{i}")

            fig2 = px.box(df,y=col,title=f"{col} Boxplot")

            st.plotly_chart(fig2,use_container_width=True,key=f"box{i}")

# ---------------- DRAG GRAPH BUILDER ----------------

    st.subheader("📊 Custom Graph Builder")

    x = st.selectbox("Select X axis",df.columns)
    y = st.selectbox("Select Y axis",df.columns)

    if chart_type == "Scatter":

        fig = px.scatter(df,x=x,y=y)

    elif chart_type == "Line":

        fig = px.line(df,x=x,y=y)

    elif chart_type == "Bar":

        fig = px.bar(df,x=x,y=y)

    elif chart_type == "Histogram":

        fig = px.histogram(df,x=x)

    elif chart_type == "3D Scatter" and len(numeric_cols)>=3:

        z = st.selectbox("Select Z axis",numeric_cols)

        fig = px.scatter_3d(df,x=x,y=y,z=z,color=z)

    st.plotly_chart(fig,use_container_width=True,key="custom_chart")

# ---------------- FLOATING CHARTS ----------------

    col1, col2 = st.columns(2)

    with col1:

        fig1 = px.histogram(df, x=df.columns[0])

        st.plotly_chart(fig1, use_container_width=True, key="hist_chart")

    with col2:

        fig2 = px.box(df, y=df.columns[1])

        st.plotly_chart(fig2, use_container_width=True, key="box_chart")

# ---------------- AI INSIGHTS ----------------

    st.subheader("🧠 AI Insights")

    insights = generate_insights(df)

    for i in insights:
        st.info(i)

# ---------------- MACHINE LEARNING ----------------

    st.subheader("🤖 Machine Learning")

    target = st.selectbox("Select Target Column",df.columns)

    if st.button("Train Models"):

        best_model,results = train_models(df,target)

        st.success(f"Best Model: {best_model}")

        res_df = pd.DataFrame({
        "Model":list(results.keys()),
        "Score":list(results.values())
        })

        st.bar_chart(res_df.set_index("Model"))

        feature_importance(df,target)

        insights = generate_insights(df)

        pdf = generate_report(df, results, insights)

        with open(pdf,"rb") as f:

            st.download_button(
            "📄 Download PDF Report",
            f,
            file_name="analysis_report.pdf"
            )

# ---------------- CHAT ----------------

    st.subheader("💬 Chat with Dataset")

    q = st.text_input("Ask question")

    if q:

        ans = chat_with_data(df,q)

        st.write(ans)
    if q:

        ans = chat_with_data(df,q)

        st.write(ans)

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
"""
<div style='text-align:center; font-size:18px; color:white; opacity:0.8'>
🚀 Made by <b>Harshit Sharma</b>
</div>
""",
unsafe_allow_html=True
)