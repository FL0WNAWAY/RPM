# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# from scipy.signal import savgol_filter
# from scipy.stats import linregress

# Streamlit app
st.title("VXP file analysis app")

# Add a file uploader widget
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Display file details
    st.write(f"You selected: {uploaded_file.name}")

    # Define column names and data types
    column_names = ['Amplitude', 'Phase', 'Time', 'Valid', 'TTLin', 'Mark', 'TTLout']
    dtype_dict = {
        'Amplitude': 'float64',
        'Phase': 'float64',
        'Time': 'int64',
        'Valid': 'int64',
        'TTLin': 'int64',
        'Mark': 'str',
        'TTLout': 'int64',
    }

    # Read CSV
    data = pd.read_csv(uploaded_file, header=None, skiprows=10, names=column_names, dtype=dtype_dict)
    st.write("### Preview of Uploaded Data")
    st.dataframe(data.head())

    # Identify rows where beam is on
    rows_beam_on = np.where((data['Time'] >= 136000) & (data['Time'] <= 205000))
    data_beam_on = data.iloc[rows_beam_on[0], :].reset_index(drop=True)

    # Convert time from ms to s
    time = data_beam_on['Time'] / 1000

    # Correct amplitude sign
    amp = -data_beam_on['Amplitude']

    # Convert phase from radians to percentile
    phase = data_beam_on['Phase'] / np.pi / 2 * 100

    # Correct for baseline drift
    trough = np.where(data_beam_on['Mark'] == 'P')
    p = np.polyfit(time[trough[0]].astype(float), amp[trough[0]].astype(float), 1)
    baseline_corr = np.polyval(p, time)
    amp_corr = amp - baseline_corr

    # Plot corrected trace
    st.write("### Amplitude Corrected for Baseline Drift")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=amp_corr, mode='lines', name='Amplitude', line=dict(color='blue')))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"), annotation_text="Baseline", annotation_position="bottom right")
   
    # Update layout
    fig.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (cm)",
    showlegend=True
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    
else:
    st.write("No file selected.")
