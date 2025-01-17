import altair as alt
import pandas as pd
import streamlit as st

st.write("""
# Experiments Results
""")

datasets = (
    "CWRU 12000",
    "CWRU 48000",
    "IMS",
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
    options=["Biased Usual", "Biased Mirrored", "Unbiased"],
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
    bias = option_bias.lower().replace(' ', '_')
    metric = "test_" + option_metric.lower().replace(' ', '_')

    indexes = []
    results = []
    std_dv = []

    # 1NN
    csv_path = ("./results/ml/1nn/" + bias + "/" + dataset + ".csv")
    df_dl = pd.read_csv(csv_path)
    indexes.append("(1) 1NN")
    results.append(df_dl[metric].mean())
    std_dv.append(df_dl[metric].std())

    # Random Forest
    csv_path = ("./results/ml/randomforest/" + bias + "/" + dataset + ".csv")
    df_dl = pd.read_csv(csv_path)
    indexes.append("(2) RF")
    results.append(df_dl[metric].mean())
    std_dv.append(df_dl[metric].std())

    # Deep Learning without interpolation
    csv_path = ("./results/deep/default/" + bias + "/results_" + dataset + ".csv")
    df_dl = pd.read_csv(csv_path)
    indexes.append("(3) CNN Default")
    results.append(df_dl[metric].mean())
    std_dv.append(df_dl[metric].std())

    try:
        # Deep Learning with interpolation
        csv_path = ("./results/deep/interpolation/results_" + dataset + ".csv")
        df_dl = pd.read_csv(csv_path)
        indexes.append("(4) CNN Interpolated")
        results.append(df_dl[metric].mean())
        std_dv.append(df_dl[metric].std())
    except FileNotFoundError:
        indexes.append("(4) CNN Interpolated")
        results.append(0)
        std_dv.append(0)

    df = pd.DataFrame({
        'Classifier': indexes,
        'Metric': results,
        'Standard Deviation': std_dv
    })

    # Interactive bar chart with altair
    chart = alt.Chart(df).mark_bar().encode(
        x='Classifier',
        y=alt.Y('Metric', scale=alt.Scale(domain=[0, 1])),
        tooltip=['Classifier', 'Metric', 'Standard Deviation']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
