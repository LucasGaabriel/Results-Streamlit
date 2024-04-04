import json

import pandas as pd
import streamlit as st

st.write("""
# Experiment Results
""")

datasets = (
    "CWRU 12000",
    "CWRU 48000",
    "EAS",
    "IMS",
    "MAFAULDA",
    "MFPT 48828",
    "MFPT 97656",
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
    options=["Biased", "Unbiased"],
    index=None,
    horizontal=True
)

option_metric = st.radio(
    "Which metric do you want to show?",
    options=["Accuracy", "Balanced Accuracy", "F1 Macro"],
    index=None,
    horizontal=True
)

if option_datasets and option_bias and option_metric:

    dataset = option_datasets.lower().replace(' ', '_')
    bias = option_bias.lower()
    metric = "test_" + option_metric.lower().replace(' ', '_')
    results = []

    # Machine Learning
    json_path = ("./results/ml/" + bias + "/" + dataset + ".json")
    result = json.load(open(json_path))
    data = {'test_accuracy': result['accuracy'],
            'test_balanced_accuracy': result['balanced accuracy'],
            'test_f1_macro': result['macro avg']['f1-score']}
    results.append(data[metric])

    # Deep Learning without interpolation
    csv_path = ("./results/deep/default/" + bias + "/results_" + dataset + ".csv")
    df_dl = pd.read_csv(csv_path)
    results.append(df_dl[metric].mean())

    # Deep Learning with interpolation
    csv_path = ("./results/deep/interpolation/results_" + dataset + ".csv")
    df_dl = pd.read_csv(csv_path)
    results.append(df_dl[metric].mean())

    index = ["(1) RF", "(2) CNN Default", "(3) CNN Interpolated"]
    df = pd.DataFrame(results, index=index, columns=["Results"])
    st.bar_chart(df)
