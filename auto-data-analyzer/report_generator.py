from fpdf import FPDF
import pandas as pd

def generate_report(df, results, insights):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"AI Data Analysis Report",ln=True)

# ---------------- DATASET SUMMARY ----------------

    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"Dataset Summary",ln=True)

    pdf.set_font("Arial","",12)

    pdf.cell(0,8,f"Rows: {df.shape[0]}",ln=True)
    pdf.cell(0,8,f"Columns: {df.shape[1]}",ln=True)

    missing = df.isnull().sum().sum()

    pdf.cell(0,8,f"Missing Values: {missing}",ln=True)

# ---------------- COLUMN INFO ----------------

    pdf.ln(5)

    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"Columns",ln=True)

    pdf.set_font("Arial","",11)

    for col in df.columns:

        pdf.cell(0,6,f"- {col}",ln=True)

# ---------------- AI INSIGHTS ----------------

    pdf.ln(5)

    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"AI Insights",ln=True)

    pdf.set_font("Arial","",11)

    for i in insights:

        pdf.multi_cell(0,6,f"- {i}")

# ---------------- MODEL RESULTS ----------------

    pdf.ln(5)

    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"Model Performance",ln=True)

    pdf.set_font("Arial","",11)

    for model,score in results.items():

        pdf.cell(0,6,f"{model}: {score}",ln=True)

# ---------------- NUMERIC SUMMARY ----------------

    pdf.ln(5)

    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"Statistical Summary",ln=True)

    summary = df.describe()

    pdf.set_font("Arial","",10)

    for col in summary.columns:

        pdf.cell(0,6,f"{col} mean: {summary[col]['mean']:.2f}",ln=True)

# ---------------- SAVE ----------------

    file = "analysis_report.pdf"

    pdf.output(file)

    return file