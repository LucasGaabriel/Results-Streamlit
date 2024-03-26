import pandas as pd
import streamlit as st

st.write("""
# Experiment Results
""")

datasets = ("CWRU", "EAS", "IMS", "MAFAULDA", "MFPT", "PU", "UOC", "XJTU")

# df = pd.read_csv("results.csv")

option = st.selectbox(
   "Which dataset would you like to show?",
   datasets,
   index=None,
   placeholder="Select a dataset."
)

