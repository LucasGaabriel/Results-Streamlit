import pandas as pd
import streamlit as st

st.write("""
# Experiment Results
""")

datasets = (
    "CWRU_12000",
    "CWRU_48000",
    "EAS",
    "IMS",
    "MAFAULDA",
    "MFPT_48828",
    "MFPT_97656",
    "PU",
    "UOC",
    "XJTU",
)

option_datasets = st.selectbox(
    "Which dataset would you like to show?",
    datasets,
    index=None,
    placeholder="Select a dataset."
)

option_bias = st.radio(
    "Do you want to show the Biased or Unbiased results?",
    key="visibility",
    options=["Biased", "Unbiased"],
    index=None,
    horizontal=True
)

if option_datasets and option_bias:
    csv_path = "./results/" + option_bias.lower() + "/results_" + option_datasets.lower() + ".csv"
    df = pd.read_csv(csv_path)
    df = df.drop(['fit_time', 'score_time'], axis=1)
    st.bar_chart(df)
