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

classifiers = (
    "Random Forest",
    "Neural Network: xresnet",
)

option_datasets = st.selectbox(
    "Which dataset would you like to show?",
    datasets,
    index=None,
    placeholder="Select a dataset."
)

option_classifier = st.selectbox(
    "Which classifier would you like to show?",
    classifiers,
    index=None,
    placeholder="Select a classifier."
)

option_interpol = st.radio(
    "Must the samples be interpolated?",
    key='interpol',
    options=["Default", "Interpolation"],
    index=None,
    horizontal=True
)

# disable_option_bias = True if option_interpol == "Interpolation" else False

option_bias = st.radio(
    "Do you want to show the Biased or Unbiased results?",
    key='bias',
    # disabled=disable_option_bias,
    options=["Biased", "Unbiased"],
    index=None,
    horizontal=True
)

if option_datasets and option_classifier and option_bias:

    if "Random Forest" in option_classifier:
            json_path = ("./results/ml/" + option_bias.lower() + "/" + option_datasets + ".json")
            result = json.load(open(json_path))
            data = {'test_accuracy': result['accuracy'],
                    'test_balanced_accuracy': result['balanced accuracy'],
                    'test_f1_macro': result['macro avg']['f1-score']}
            df = pd.DataFrame(data, index=[0])

    if "Neural Network" in option_classifier:
        if "Interpolation" in option_interpol:
            csv_path = ("./results/deep/" + option_interpol.lower() + "/"
                        + "/results_" + option_datasets.lower() + ".csv")
        else:
            csv_path = ("./results/deep/" + option_interpol.lower() + "/" + option_bias.lower()
                        + "/results_" + option_datasets.lower() + ".csv")

        df = pd.read_csv(csv_path)
        if 'fold' in df.columns.tolist():
            df = df.drop(['fold'], axis=1)
        df = df.drop(['fit_time', 'score_time'], axis=1)

    st.bar_chart(df)


