import streamlit as st

# Streamlit app
st.title("File Selector App")

# Add a file uploader widget
uploaded_file = st.file_uploader("Choose a file")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Display file details
    st.write(f"You selected: {uploaded_file.name}")
    # Optionally, read and display the file contents (e.g., CSV)
    if uploaded_file.name.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
else:
    st.write("No file selected.")