#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import sys
from datetime import datetime as dt, timedelta
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def upload_excel():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            birth = pd.read_csv(uploaded_file)
            
            # Convert 'DOB' column to datetime format
            birth['DOB'] = pd.to_datetime(birth['DOB'], errors='coerce')
            
            st.success("File uploaded successfully. DataFrame 'birth' created.")
            return birth
        except Exception as e:
            st.error(f"Error: {e}")
    return None
def main():
    st.title("Streamlit App with CSV Upload")

    # Call the function to upload CSV file and get DataFrame
    birth = upload_excel()

    # Rest of your Streamlit app logic goes here
    if birth is not None:
        st.write(f"Data types of columns in 'birth': {birth.dtypes}")

        # Convert 'DOB' column to datetime, if it's not already
        if 'DOB' in birth.columns and pd.api.types.is_datetime64_any_dtype(birth['DOB']):
            birth['DOB'] = birth['DOB'].dt.year
            birth['Name'] = birth['Name'].str.split('Last').str[1].str.split(', First').str.join('-')
            birth = birth[birth['DOB'].notna()]
            birth = birth.sort_values('DOB', ascending=True)

            dup_birth = birth[birth['Name'].duplicated(keep=False)]
            dup_birth = dup_birth.sort_values('Name', ascending=True)

            st.write(f"Length of 'birth': {len(birth)}")
            st.write("Preview of 'birth' DataFrame:")
            st.write(birth.head())
        else:
            st.error("The 'DOB' column is not recognized as datetime. Check your CSV file format.")
if __name__ == "__main__":
    main()

