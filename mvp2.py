#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import sys
from datetime import datetime as dt, timedelta
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def determine_severity(score):
    if 6 <= score <= 12:
        return 'High Risk'
    elif 13 <= score <= 14:
        return 'Moderate Risk'
    elif 15 <= score <= 18:
        return 'Low Risk'
    elif 19 <= score <= 23:
        return 'Healthy'
    else:
        return 'Unknown'

def process_ulcer_data(ulcer):
    ulcer['Name'] = ulcer['Name'].str.split('Last').str[1].str.split(', First').str.join('-')

    # Change column names
    ulcer = ulcer.rename(columns={ulcer.columns[2]: 'Location'})
    ulcer = ulcer.rename(columns={ulcer.columns[5]: 'Activated'})
    
    # Modify the Type column
    roman_to_arabic = {'I': '1', 'II': '2', 'III': '3', 'IV': '4'}
    ulcer['Type'] = ulcer['Type'].str.extract('STAGE (\w+)')[0].map(roman_to_arabic)
    ulcer['SOE'] = pd.to_datetime(ulcer['SOE'])
    ulcer['Onset'] = pd.to_datetime(ulcer['Onset'])
    ulcer['Activated'] = pd.to_datetime(ulcer['Activated'])
    
    # ulcer = ulcer.sort_values('SOE' , ascending=True)
    ulcer["Type"] = ulcer["Type"].astype(int)

def process_brad_data(brad):
    brad['Severity'] = brad['AssessmentAnswer'].apply(determine_severity)
    # Strip the Name column to remove 'Last' and 'First' and retain the numbers
    brad['Name'] = brad['Name'].str.split('Last').str[1].str.split(', First').str.join('-')
    brad['Visitdate'] = pd.to_datetime(brad['Visitdate'])
    brad['Worker_name'] = brad['Textbox65'].str.split(':').str[1].str.split(',').str[0]
    brad['Worker_type'] = brad['Textbox65'].str.rsplit(',').str[1]

def upload_ulcer_csv():
    uploaded_file = st.file_uploader("Choose a CSV file for Ulcer dataset", type="csv")

    if uploaded_file is not None:
        try:
            ulcer = pd.read_csv(uploaded_file)
            st.success("Ulcer dataset uploaded successfully.")
            return ulcer
        except Exception as e:
            st.error(f"Error: {e}")
    return None

def upload_brad_csv():
    uploaded_file = st.file_uploader("Choose a CSV file for Physical Assessment dataset", type="csv")

    if uploaded_file is not None:
        try:
            brad = pd.read_csv(uploaded_file)
            st.success("Physical Assessment dataset uploaded successfully.")
            return brad
        except Exception as e:
            st.error(f"Error: {e}")
    return None

def merge_and_process_data(ulcer, brad):
    # Merge two tables
    ulcer_b = ulcer.merge(brad, left_on='Name', right_on='Name')
    ulcer_b = ulcer_b.dropna()

    # Ensure 'SOE' column is in datetime format
    ulcer_b['SOE'] = pd.to_datetime(ulcer_b['SOE'], errors='coerce')

    # Calculate the maximum allowed date (SOE Date + 60 days)
    ulcer_b['MaxAllowedDate'] = ulcer_b['SOE'] + pd.to_timedelta(60, unit='D')

    # Filter the rows where VisitDate is greater than SOEDate but not greater than 60 days
    ulcer_b = ulcer_b[(ulcer_b['Visitdate'] >= ulcer_b['SOE']) & (ulcer_b['Visitdate'] <= ulcer_b['MaxAllowedDate'])]

    # Drop the MaxAllowedDate column
    ulcer_b = ulcer_b.drop(columns=['MaxAllowedDate'])
    ulcer_b = ulcer_b.sort_values('Name', ascending=True)

    return ulcer_b  # Return the processed DataFrame
    
def main():
    st.title("Streamlit App with CSV Upload")

    # Call the function to upload CSV file for Ulcer dataset
    ulcer = upload_ulcer_csv()

    # Call the function to upload CSV file for Brad dataset
    brad = upload_brad_csv()

    # Display the processed Ulcer dataset
    if ulcer is not None:

        process_ulcer_data(ulcer)
        st.write(f"Length of 'Ulcer Data': {len(ulcer)}")
        st.write("Preview of 'Ulcer Data' DataFrame:")
        st.write(ulcer.head(10))

    # Display the processed brad dataset
    if brad is not None:

        process_brad_data(brad)
        st.write(f"Length of 'brad Data': {len(brad)}")
        st.write("Preview of 'brad Data' DataFrame:")
        st.write(brad.head(10))

    # Merge and process data
    if ulcer is not None and brad is not None:
        ulcer_b = merge_and_process_data(ulcer, brad)  # Get the processed DataFrame
        st.write("Processed 'Ulcer Data' DataFrame:")
        st.write(ulcer_b.head(10))
        
if __name__ == "__main__":
    main()

