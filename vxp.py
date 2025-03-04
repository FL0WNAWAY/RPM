import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Streamlit app
st.title("Breathing Trace Analysis")

# Add a file uploader widget
uploaded_file = st.file_uploader("Upload the trace csv file", type=["csv"])

if uploaded_file is not None:
    # Display file details
    # st.write(f"You selected: {uploaded_file.name}")

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
    st.write("### Beam-On Amplitude Corrected for Baseline Drift")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=amp_corr, mode='lines', name='Amplitude', line=dict(color='blue')))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"), annotation_text="Baseline", annotation_position="bottom right")
   
    # Update layout
    fig.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (cm)",
    showlegend=False
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Convert phase threshold to amplitude threshold
    # Ask the user to input phase upper and lower thresholds
    ph_max = st.number_input("Enter the upper phase threshold (e.g. 75):", value=75, step=10)
    ph_min = st.number_input("Enter the lower phase threshold (e.g. 25):", value=25, step=10)
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
    
    # Format the columns to 1 decimal place
    result.iloc[:, 1:] = result.iloc[:, 1:].applymap(lambda x: f"{x:.1f}")

    st.write("### Amplitude Threshold Data")
    st.table(result)

    # Ask the user to input amplitude threshold
    amp_max_mean = (float(result.iloc[0, 1]) + float(result.iloc[1, 1])) / 2
    amp_max_mean = round(amp_max_mean, 1)
    st.write(amp_max_mean)
    amp_max = st.number_input("Enter the upper amplitude threshold (mm):", value=0, step=0.1)
    amp_min = st.number_input("Enter the lower amplitude threshold (mm):", value=-0.1, step=0.1)
    
   # Calculate duty cycle
    amp_max, amp_min = 0.2, -0.1
    rows_in_thres = np.where((amp_corr > amp_min) & (amp_corr < amp_max))[0]
    duty_cycle = round(len(rows_in_thres) / len(amp_corr) * 100, 0)

    st.write(f"### Duty Cycle: **{duty_cycle}%**")

    # plot corrected trace with gating threshold
    fig = go.Figure()

    # Add the corrected amplitude trace
    fig.add_trace(go.Scatter(x=time, y=amp_corr, mode='lines', name='Corrected Amplitude', line=dict(color='blue')))

    # Add horizontal lines for baseline and gating thresholds
    fig.add_hline(y=0, line=dict(color='black', width=2), name="Baseline")
    fig.add_hline(y=amp_min, line=dict(color='green', width=2, dash='dash'), name="Gating Threshold (Min)")
    fig.add_hline(y=amp_max, line=dict(color='green', width=2, dash='dash'), name="Gating Threshold (Max)")

    # Update layout
    fig.update_layout(
    title="Corrected Trace with Gating Threshold",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (cm)",
    legend_title="Legend",
    template="plotly_white"
    )

    # Display the figure in Streamlit
    st.write("### Corrected Trace with Gating Threshold")
    st.plotly_chart(fig)
else:
    st.write("No file selected.")
