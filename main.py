import json

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

    df = pd.DataFrame()
    metric = "test_" + option_metric.lower().replace(' ', '_')

    # Machine Learning
    json_path = ("./results/ml/" + option_bias.lower() + "/" + option_datasets + ".json")
    result = json.load(open(json_path))
    data = {'test_accuracy': result['accuracy'],
            'test_balanced_accuracy': result['balanced accuracy'],
            'test_f1_macro': result['macro avg']['f1-score']}
    df.insert(0, "ml", pd.Series(data[metric]))

    # Deep Learning without interpolation
    csv_path = ("./results/deep/default/" + option_bias.lower() + "/results_" +
                option_datasets.lower() + ".csv")
    df_dl = pd.read_csv(csv_path)
    df.insert(1, "dl_default", pd.Series(df_dl[metric].mean()))
    # df = pd.concat([df, pd.Series(df_dl[metric].mean())], ignore_index=True)

    # Deep Learning with interpolation
    csv_path = ("./results/deep/interpolation/results_" + option_datasets.lower() + ".csv")
    df_dl = pd.read_csv(csv_path)
    df.insert(2, "dl_interpolation", pd.Series(df_dl[metric].mean()))

    st.bar_chart(df)
