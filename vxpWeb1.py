
import numpy as np
import pandas as pd
import streamlit as st
# import plotly.graph_objects as go
import matplotlib.pyplot as plt

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

    # Convert time from ms to s
    data['Time'] = data['Time'] / 1000

    # Correct amplitude sign
    data['Amplitude'] = -data['Amplitude']

    # Convert phase from radians to percentile
    data['Phase'] = data['Phase'] / np.pi / 2 * 100    
     
    # Identify rows where beam is on
    if 0 not in data['TTLin'].values:
        print ("none")
        t1 = st.number_input("Enter the start time of 4DCT (s):", value=0, step=1)
        t2 = st.number_input("Enter the finish time of 4DCT (s):", value=0, step=1)
        rows_beam_on = np.where((data['Time'] >= t1) & (data['Time'] <= t2))

    else:
        rows_beam_on = np.where(data['TTLin'] == 0)
    data_beam_on = data.iloc[rows_beam_on[0], :].reset_index(drop=True)

    # Display preview of beam on data
    #st.write("### Preview of Uploaded Data")
    #st.dataframe(data_beam_on.head())

    # Checkbox for baseline drift correction / show 0 phase / show 50 phase
    col1, col2, col3 = st.columns(3)
    with col1:
        apply_corr = st.checkbox("Correct for baseline")
    with col2:
        show0 = st.checkbox("Check peaks / 0% phase")
    with col3:
        show50 = st.checkbox("Check troughs / 50% phase")       


    # Correct for baseline drift
    trough = np.where(data_beam_on['Mark'] == 'P')
    p = np.polyfit(data_beam_on['Time'][trough[0]].astype(float), data_beam_on['Amplitude'][trough[0]].astype(float), 1)
    baseline_corr = np.polyval(p, data_beam_on['Time'])
    
    # only apply correction is baseline drift is downwards
    #if p[0]<0:
    #    amp_corr = data_beam_on['Amplitude'] - baseline_corr
    #else:
    #    amp_corr = data_beam_on['Amplitude'] - np.mean(data_beam_on['Amplitude'][trough[0]])

    # locate 0 and 50 phases
    ppeak = np.zeros(np.size(data_beam_on['Time']))
    ptrough = np.zeros(np.size(data_beam_on['Time']))
    rows_ph_peak = np.where(data_beam_on['Phase'] == 0)[0]
    rows_ph_trough = np.where(np.abs(data_beam_on['Phase'] - 50) < 1)[0]
    ppeak[rows_ph_peak] = 1
    ptrough[rows_ph_trough] = 1

    # Option to find tune amplitude adjustment
    #tune = st.number_input("Apply shift to the amplitude in mm:", value=0.0, step=0.1)
    if "button1_clicks" not in st.session_state:
        st.session_state.button1_clicks = 0
    if "button2_clicks" not in st.session_state:
        st.session_state.button2_clicks = 0

    col_up, col_down, col_reset = st.columns(3)
    with col_up:
        if st.button("Shift the curve up by 0.1mm"):
            st.session_state.button1_clicks += 1
    with col_down:
        if st.button("Shift the curve down by 0.1mm"):
            st.session_state.button2_clicks += 1
    with col_reset:
        if st.button("Reset"):
            st.session_state.button1_clicks = 0
            st.session_state.button2_clicks = 0
    
    # Plot beam-on trace
    st.write("### Breathing trace during 4DCT acquisition")
    
    fig, ax = plt.subplots()
    
    if apply_corr:
        amp_corr = data_beam_on['Amplitude'] - baseline_corr
    else:
        amp_corr = data_beam_on['Amplitude'] - np.mean(data_beam_on['Amplitude'][trough[0]])
    amp_corr = amp_corr + 0.01*(st.session_state.button1_clicks-st.session_state.button2_clicks)
    
    ax.plot(data_beam_on['Time'], amp_corr, color='blue', label='Amplitude')
    
    if show0:
        ax.plot(data_beam_on['Time'], ppeak, color='yellow', label='0% phase')
    if show50:   
        ax.plot(data_beam_on['Time'], ptrough, color='cyan', label='50% phase')
    
    ax.axhline(y=0, color='black', linestyle='dashed', label='Baseline')
    
    # Add labels and legend
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (cm)")
    ax.legend(loc="best")
    
    # Display the plot in Streamlit
    st.pyplot(fig)

    # Convert phase threshold to amplitude threshold
    # Ask the user to input phase upper and lower thresholds
    ph_max = st.number_input("Enter the upper phase threshold (e.g. 75):", value=75, step=10)
    ph_min = st.number_input("Enter the lower phase threshold (e.g. 25):", value=25, step=10)
    rows_ph_max = np.where(np.abs(data_beam_on['Phase'] - ph_max) < 1)[0]
    rows_ph_min = np.where(np.abs(data_beam_on['Phase'] - ph_min) < 1)[0]
    amp_ph_max = amp_corr[rows_ph_max]
    amp_ph_min = amp_corr[rows_ph_min]
    
    # Plot selected upper and lower phase
    #pmax = np.zeros(np.size(data_beam_on['Time']));
    #pmin = np.zeros(np.size(data_beam_on['Time']));
    #pmax[rows_ph_max] = 1
    #pmin[rows_ph_min] = 1
    
    #st.write("### Selected upper and lower phase threshold")
    
    #fig, ax = plt.subplots()
    #ax.plot(data_beam_on['Time'], amp_corr, color='blue', label='Amplitude')
    #ax.plot(data_beam_on['Time'], pmax, color='yellow', label='upper phase threshold')
    #ax.plot(data_beam_on['Time'], pmin, color='cyan', label='lower phase threshold')
      
    # Add labels and legend
    #ax.set_xlabel("Time")
    #ax.set_ylabel("Amplitude")
    #ax.legend(loc="best")
    
    # Display the plot in Streamlit
    #st.pyplot(fig)

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
    #result.iloc[:, 1:] = result.iloc[:, 1:].map(lambda x: f"{x:.1f}")
    result.iloc[:, 1:] = result.iloc[:, 1:].applymap(lambda x: f"{x:.1f}")

    st.write("### Amplitude Threshold Data")
    st.table(result)

    # Ask the user to input amplitude threshold
    amp_max_mean = (float(result.iloc[0, 1]) + float(result.iloc[1, 1])) / 2
    amp_max_mean = round(amp_max_mean, 1)
    amp_max = st.number_input("Enter the upper amplitude threshold in cm:", value=amp_max_mean, step=0.1,key="upper_amp")
    amp_min = st.number_input("Enter the lower amplitude threshold in cm:", value=-0.1, step=0.1,key="lower_amp")
    
   # Calculate duty cycle
    rows_in_thres = np.where((amp_corr > amp_min) & (amp_corr < amp_max))[0]
    duty_cycle = round(len(rows_in_thres) / len(amp_corr) * 100, 0)

    st.write(f"### Duty Cycle: **{duty_cycle}%**")

    # plot trace with gating threshold
     
    fig, ax = plt.subplots()
    ax.plot(data_beam_on['Time'], amp_corr, color='blue', label='Amplitude')
    ax.axhline(y=amp_min, color='green', label='Gating Threshold (Min)')
    ax.axhline(y=amp_max, color='green', label='Gating Threshold (Max)')   
    ax.axhline(y=0, color='black', linestyle='dashed', label='Baseline') 

    # Add labels and legend
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (cm)")
    ax.legend(loc="best")
    
    # Display the plot in Streamlit
    st.pyplot(fig)

else:
    st.write("No file selected.")
