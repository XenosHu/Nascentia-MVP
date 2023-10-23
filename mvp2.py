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
    # Rename columns containing 'Loca' to 'Location'
    location_columns = ulcer.filter(like='Loca')
    location_columns = {col: 'Location' for col in location_columns.columns}
    ulcer = ulcer.rename(columns=location_columns)

    # Rename columns containing 'Acti' to 'Activated'
    activated_columns = ulcer.filter(like='Acti')
    activated_columns = {col: 'Activated' for col in activated_columns.columns}
    ulcer = ulcer.rename(columns=activated_columns)
    
    # Modify the Type column
    type_mapping = {
        'PRESSURE ULCER STAGE I': 1,
        'PRESSURE ULCER STAGE II': 2,
        'PRESSURE ULCER STAGE III': 3,
        'PRESSURE ULCER STAGE IV': 4,
    }
    ulcer['Type'] = ulcer['Type'].map(type_mapping)
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

def plot_patient_data(patient_id, brad):
    # Filter data for the specified patient
    sub = brad[brad['Name'] == patient_id]
    sub = sub.sort_values('Visitdate', ascending=True)

    # Plot line chart
    plt.figure(figsize=(10, 6))
    plt.plot(sub['Visitdate'], sub['AssessmentAnswer'], marker='o', linestyle='-')
    plt.title(f'Assessment Braden score of Patient {patient_id} over Time')
    plt.xlabel('Visit Date')
    plt.ylabel('Assessment Answer')
    plt.tight_layout()

    # Display the plot using Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def plot_ulcer_counts(ulcer):
    # Create a filtered DataFrame with unique patients, keeping the most recent record
    unique_ulcer_patients = ulcer.sort_values('SOE', ascending=False).drop_duplicates('Name')

    # Plot bar chart for Pressure Ulcer Count by Type
    type_counts = unique_ulcer_patients['Type'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar', title='Pressure Ulcer Count by Type', color='skyblue')
    plt.xlabel('Type')
    plt.ylabel('Count')

    # Add labels for each bar
    for i, count in enumerate(type_counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.xticks(rotation=0, ha='left')
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot()


def plot_severity_counts(brad):
    
    unique_brad_patients = brad.sort_values('Visitdate', ascending=False).drop_duplicates('Name')
    type_counts = unique_brad_patients['Severity'].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar', title='Severity levels count over all', color='skyblue')
    plt.xlabel('Severity')
    plt.ylabel('Count')

    # Add labels for each bar
    for i, count in enumerate(type_counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.xticks(rotation=0, ha='left')
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot()
    
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
        #st.write("Preview of 'Ulcer Data' DataFrame:")
        #st.write(ulcer.head(10))

    # Display the processed brad dataset
    if brad is not None:

        process_brad_data(brad)
        st.write(f"Length of 'brad Data': {len(brad)}")
        #st.write("Preview of 'brad Data' DataFrame:")
        #st.write(brad.head(10))

    # Merge and process data
    if ulcer is not None and brad is not None:
        ulcer_b = merge_and_process_data(ulcer, brad)  # Get the processed DataFrame
        st.write(f"Length of 'ulcer merge physical assignment Data': {len(ulcer_b)}")
        st.write("Preview of 'ulcer merge physical assignment Data' DataFrame:")
        st.write(ulcer_b.head(10))

        # Allow user to input a patient ID
        patient_id = st.text_input("Enter Patient ID (in format of First-Last, e.g. 12-345) for their Braden score history:")
        
        # Check if the patient ID is provided
        if patient_id:
            # Plot line chart for the specified patient
            plot_patient_data(patient_id, brad)

    if brad is not None:
        plot_severity_counts(brad)
    if ulcer is not None and brad is not None:
        plot_ulcer_counts(ulcer_b)

        
if __name__ == "__main__":
    main()

