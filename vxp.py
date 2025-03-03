# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# %%
# Ensure required packages are installed
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    from scipy.stats import linregress
except ModuleNotFoundError as e:
    print(f"Module not found: {e.name}. Please install missing packages.")
    exit()

# %%
# Load breathing trace
Tk().withdraw()  # Hide main window
file_path = askopenfilename(filetypes=[("CSV files", "*.csv")], title="Select file")
if not file_path:
    print("No file selected. Exiting.")
    exit()

# Define column data types
column_names = ['Amplitude', 'Phase', 'Time', 'Valid', 'TTLin', 'Mark', 'TTLout']
dtype_dict = {
    'Amplitude': 'float64',   # amp
    'Phase': 'float64',   # phase
    'Time': 'int64',   # time
    'Valid': 'int64',   # valid flag
    'TTLin': 'int64',   # ttlin, 0 for beam-on on Siemens CT
    'Mark': 'str',    # mark P for trough, Z for peak
    'TTLout': 'int64',   # ttlout, always 0
}

data = pd.read_csv(file_path, header=None, skiprows=10, names=column_names, dtype=dtype_dict)  # Read CSV file into NumPy array
#data = pd.read_csv(file_path, skiprows=10, header=None, usecols=[0, 1, 2, 4])
# %%
# Identify rows where beam is on
rows_beam_on = np.where(data.iloc[:, 4] == 0)[0]

# for RKH GE scanners where beam on time needs to be mannually recorded
rows_beam_on = np.where((data.iloc[:, 2] >= 136000) & (data.iloc[:, 2] <= 205000))

# get beam on data
data_beam_on = data.iloc[rows_beam_on[0], :].reset_index(drop=True)

# Convert time unit from ms to s
time = data_beam_on.iloc[:, 2] / 1000

# Correct amplitude sign
amp = -data_beam_on.iloc[:, 0]

# Convert phase unit from radians to percentile
phase = data_beam_on.iloc[:, 1] / np.pi / 2 * 100

# Smooth data using Savitzky-Golay filter
amp_sm = savgol_filter(amp, window_length=11, polyorder=3)

# Plot original vs smoothed data
plt.figure()
plt.plot(time, amp, 'r--', label='Original')
plt.plot(time, amp_sm, 'b', label='Smoothed')
plt.legend()
plt.show()

# Correct for baseline drift
trough = np.where(data_beam_on.iloc[:,5] == 'P');
a=time[trough[0]]
p = np.polyfit(time[trough[0]].astype(float), amp_sm[trough[0]].astype(float), 1)
f = np.polyval(p, time[trough[0]])

# Plot identified minima and linear fit
plt.figure()
plt.scatter(time[trough[0]], amp_sm[trough[0]], c='r', label='Trough')
plt.plot(time[trough[0]], f, 'b', label='Linear Fit')
plt.legend()
plt.show()

# Correct the trace using the linear model
baseline_corr = np.polyval(p, time)
amp_corr = amp_sm - baseline_corr

# Plot corrected trace
plt.figure()
plt.plot(time, amp_corr, 'b')
plt.axhline(0, color='k', linestyle='-', label='Baseline')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (cm)')
plt.title('Smoothed Amplitude Corrected for Baseline Drift')
plt.legend()
plt.show()

# Convert phase threshold to amplitude threshold
ph_max, ph_min = 75, 25
rows_ph_max = np.where(np.abs(phase - ph_max) < 1)[0]
rows_ph_min = np.where(np.abs(phase - ph_min) < 1)[0]
amp_ph_max = amp_corr[rows_ph_max]
amp_ph_min = amp_corr[rows_ph_min]

# Store results in a DataFrame
result = pd.DataFrame({
    "Amplitude Threshold": ["Upper Phase Amplitude", "Lower Phase Amplitude"],
    "Mean (cm)": [np.mean(amp_ph_max), np.mean(amp_ph_min)],
    "Max (cm)": [np.max(amp_ph_max), np.max(amp_ph_min)],
    "Min (cm)": [np.min(amp_ph_max), np.min(amp_ph_min)],
    "Std Dev (cm)": [np.std(amp_ph_max), np.std(amp_ph_min)]
})
result.iloc[:, 1:] = result.iloc[:, 1:].round(1)
print(result)

# Calculate duty cycle
amp_max, amp_min = 0.2, -0.1
rows_in_thres = np.where((amp_corr > amp_min) & (amp_corr < amp_max))[0]
duty_cycle = round(len(rows_in_thres) / len(amp_corr) * 100, 0)
print(f"Duty Cycle: {duty_cycle}%")

# Plot corrected trace with gating threshold
plt.figure()
plt.plot(time, amp_corr, 'b')
plt.axhline(0, color='k', linestyle='-', label='Baseline')
plt.axhline(amp_min, color='g', linestyle='-', label='Gating Threshold')
plt.axhline(amp_max, color='g', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (cm)')
plt.title('Amplitude Corrected for Baseline Drift with Gating Threshold')
plt.legend()
plt.show()
