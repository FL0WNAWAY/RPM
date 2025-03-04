# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
# from scipy.signal import savgol_filter
#from scipy.stats import linregress

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

    # Apply Savitzky-Golay filter
    # amp_sm = savgol_filter(amp, window_length=11, polyorder=3)

    # Plot original vs smoothed data
    #st.write("### Original vs Smoothed Data")
    #fig, ax = plt.subplots()
    #ax.plot(time, amp, 'r--', label='Original')
    #ax.plot(time, amp_sm, 'b', label='Smoothed')
    #ax.legend()
    #st.pyplot(fig)

    # Correct for baseline drift
    trough = np.where(data_beam_on['Mark'] == 'P')
    p = np.polyfit(time[trough[0]].astype(float), amp_sm[trough[0]].astype(float), 1)
    baseline_corr = np.polyval(p, time)
    amp_corr = amp - baseline_corr

    # Plot corrected trace
    st.write("### Smoothed Amplitude Corrected for Baseline Drift")
    fig, ax = plt.subplots()
    ax.plot(time, amp_corr, 'b')
    ax.axhline(0, color='k', linestyle='-', label='Baseline')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (cm)')
    ax.legend()
    st.pyplot(fig)

    # Convert phase threshold to amplitude threshold
    ph_max, ph_min = 75, 25
    rows_ph_max = np.where(np.abs(phase - ph_max) < 1)[0]
    rows_ph_min = np.where(np.abs(phase - ph_min) < 1)[0]
    amp_ph_max = amp_corr[rows_ph_max]
    amp_ph_min = amp_corr[rows_ph_min]

    # Store results in DataFrame
    result = pd.DataFrame({
        "Amplitude Threshold": ["Upper Phase Amplitude", "Lower Phase Amplitude"],
        "Mean (cm)": [np.mean(amp_ph_max), np.mean(amp_ph_min)],
        "Max (cm)": [np.max(amp_ph_max), np.max(amp_ph_min)],
        "Min (cm)": [np.min(amp_ph_max), np.min(amp_ph_min)],
        "Std Dev (cm)": [np.std(amp_ph_max), np.std(amp_ph_min)]
    })
    result.iloc[:, 1:] = result.iloc[:, 1:].round(1)

    st.write("### Amplitude Threshold Data")
    st.table(result)

    # Calculate duty cycle
    amp_max, amp_min = 0.2, -0.1
    rows_in_thres = np.where((amp_corr > amp_min) & (amp_corr < amp_max))[0]
    duty_cycle = round(len(rows_in_thres) / len(amp_corr) * 100, 0)

    st.write(f"### Duty Cycle: **{duty_cycle}%**")

    # Plot corrected trace with gating threshold
    st.write("### Corrected Trace with Gating Threshold")
    fig, ax = plt.subplots()
    ax.plot(time, amp_corr, 'b')
    ax.axhline(0, color='k', linestyle='-', label='Baseline')
    ax.axhline(amp_min, color='g', linestyle='-', label='Gating Threshold')
    ax.axhline(amp_max, color='g', linestyle='-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (cm)')
    ax.legend()
    st.pyplot(fig)
    
else:
    st.write("No file selected.")
