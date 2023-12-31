#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import os
import sys
import ast
import altair as alt
from datetime import datetime as dt, timedelta
import numpy as np
import matplotlib.pyplot as plt
from plotly import colors, express as px, graph_objects as go, offline as pyo
import streamlit as st
from collections import defaultdict
import subprocess
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from PIL import Image
import torch
from torchvision import transforms
import io
import hmac

# # Install dependencies
# subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

# Execute setup.sh
#subprocess.run("bash setup.sh", shell=True, check=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_and_infer_image(uploaded_file, model):
    # Load the image
    img = Image.open(uploaded_file)

    # Preprocess the image
    img = img.resize((640, 640))
    img = np.array(img) / 255.0  # Normalize the image to [0, 1]

    # Convert the image to a PyTorch tensor
    img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
    model = model.float()

    # Perform inference
    with torch.no_grad():
        detections = model(img_tensor)

    return detections

def display_results(detections):
    #Labeling
    class_labels = ['healing', 'necrotic', 'sloughy', 'superficial', 'undermine']
    class_ids = [0, 1, 2, 3, 4]
    # Create a mapping from class IDs to class labels
    class_id_to_label = {class_id: label for class_id, label in zip(class_ids, class_labels)}

    # Get the class ID with the highest confidence from the result tensor
    predicted_class_id = detections.argmax(dim=1).item()
    # Match the class ID to the class label using the mapping
    predicted_class_label = class_id_to_label.get(predicted_class_id, 'Unknown')
    detection_result = f"{round(detections.max().item() * 100, 2)}%"
    return predicted_class_label, detection_result

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

def upload_birth_csv():
    st.write("**Choose a CSV file for Birthday dataset**")
    uploaded_file = st.file_uploader("", type="csv")

    if uploaded_file is not None:
        try:
            birth = pd.read_csv(uploaded_file)
            st.success("Birthday dataset uploaded successfully.")
            return birth
        except Exception as e:
            st.error(f"Error: {e}")
    return None

def upload_ulcer_csv():
    st.write("**Choose a CSV file for Ulcer dataset**")
    uploaded_file = st.file_uploader(" ", type="csv")

    if uploaded_file is not None:
        try:
            ulcer = pd.read_csv(uploaded_file)
            st.success("Ulcer dataset uploaded successfully.")
            return ulcer
        except Exception as e:
            st.error(f"Error: {e}")
    return None

def upload_brad_csv():
    st.write("**Choose a CSV file for Physical Assessment dataset**")
    uploaded_file = st.file_uploader("  ", type="csv")

    if uploaded_file is not None:
        try:
            brad = pd.read_csv(uploaded_file)
            st.success("Physical Assessment dataset uploaded successfully.")
            return brad
        except Exception as e:
            st.error(f"Error: {e}")
    return None

def filter_date(ulcer, brad):
    ulcer['SOE'] = pd.to_datetime(ulcer['SOE'])
    brad['Visitdate'] = pd.to_datetime(brad['Visitdate'])

    min_date = pd.to_datetime(min(ulcer['SOE'].min(), brad['Visitdate'].min()))
    max_date = pd.to_datetime(max(ulcer['SOE'].max(), brad['Visitdate'].max()))

    st.subheader("Filter Data by Dates")
    start_date = st.date_input('**Choose start date:**', min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
    end_date = st.date_input('**Choose end date:**', min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())
    
    # Convert date_range to datetime for comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter ulcer data
    ulcer = ulcer[(ulcer['SOE'] >= start_date) & (ulcer['SOE'] <= end_date)]

    # Filter brad data
    brad = brad[(brad['Visitdate'] >= start_date) & (brad['Visitdate'] <= end_date)]

    return ulcer, brad

def process_birth_data(birth):
    birth['DOB'] = pd.to_datetime(birth['DOB'])
    birth['DOB'] = birth['DOB'].dt.year
    birth['Name'] = birth['Name'].str.split('Last').str[1].str.split(', First').str.join('-')
    birth = birth[birth['DOB'].notna()]
    birth = birth.sort_values('DOB', ascending=True)

    dup_birth = birth[birth['Name'].duplicated(keep=False)]
    dup_birth = dup_birth.sort_values('Name', ascending=True)
    return birth

def process_ulcer_data(ulcer):
    ulcer['Name'] = ulcer['Name'].str.split('Last').str[1].str.split(', First').str.join('-')

    # Change column names
    # Rename columns containing 'Loca' to 'Location'
    location_columns = ulcer.filter(like='Loca')
    location_columns = {col: 'Location' for col in location_columns.columns}
    ulcer.rename(columns=location_columns, inplace=True)

    # Rename columns containing 'Acti' to 'Activated'
    activated_columns = ulcer.filter(like='Acti')
    activated_columns = {col: 'Activated' for col in activated_columns.columns}
    ulcer.rename(columns=activated_columns, inplace=True)
    
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
    
    ulcer["Type"] = ulcer["Type"].astype(int)

    return ulcer  # Return the modified DataFrame

def process_brad_data(brad):
    brad['Severity'] = brad['AssessmentAnswer'].apply(determine_severity)
    # Strip the Name column to remove 'Last' and 'First' and retain the numbers
    brad['Name'] = brad['Name'].str.split('Last').str[1].str.split(', First').str.join('-')
    brad['Visitdate'] = pd.to_datetime(brad['Visitdate'])
    brad['Worker_name'] = brad['Textbox65'].str.split(':').str[1].str.split(',').str[0]
    brad['Worker_type'] = brad['Textbox65'].str.rsplit(',').str[1]
    
# def plot_patient_data(patient_id, brad):
#     # Filter data for the specified patient
#     sub = brad[brad['Name'] == patient_id]
#     sub = sub.sort_values('Visitdate', ascending=True)

#     # Plot line chart
#     plt.figure(figsize=(10, 6))
#     plt.plot(sub['Visitdate'], sub['AssessmentAnswer'], marker='o', linestyle='-')
#     plt.title(f'Assessment Braden score of Patient {patient_id} over Time')
#     plt.xlabel('Visit Date')
#     plt.ylabel('Assessment Answer')
#     plt.tight_layout()

#     # Display the plot using Streamlit
#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     st.pyplot()

def plot_patient_data(patient_id, brad, ulcer):
    # Filter data for the specified patient in the brad dataframe
    sub_brad = brad[brad['Name'] == patient_id]
    sub_brad = sub_brad.sort_values('Visitdate', ascending=True)

    # Plot line chart for brad data
    plt.figure(figsize=(10, 6))
    plt.plot(sub_brad['Visitdate'], sub_brad['AssessmentAnswer'], marker='o', linestyle='-')
    plt.title(f'Assessment Braden score of Patient {patient_id} over Time')
    plt.xlabel('Visit Date')
    plt.ylabel('Assessment Answer')
    plt.tight_layout()

    # Display the plot using Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # Filter data for the specified patient in the ulcer dataframe
    sub_ulcer = ulcer[ulcer['Name'] == patient_id]

    # Check if there are matching rows in the ulcer dataframe
    if not sub_ulcer.empty:
        st.write(sub_ulcer)
    else:
        st.write("No matching ulcer data for this patient ID.")

def plot_ulcer_counts(ulcer):
    # Create a filtered DataFrame with unique patients, keeping the most recent record
    unique_ulcer_patients = ulcer.sort_values('SOE', ascending=False).drop_duplicates('Name')

    # Plot bar chart for Pressure Ulcer Count by Type
    type_counts = unique_ulcer_patients['Type'].value_counts().sort_index()

    colors = plt.cm.RdYlGn(np.linspace(1, 0, len(type_counts)))
    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar', title='Pressure ulcer count by Type for all unique patients', color = colors)
    plt.xlabel('Type')
    plt.ylabel('Count')

    # Add labels for each bar
    for i, count in enumerate(type_counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot()


def plot_ulcer_counts_by_month(ulcer):
    sub = ulcer.copy()
    sub['Month'] = pd.to_datetime(sub['SOE']).dt.to_period('M').astype(str)
    sub['Quarter'] = pd.to_datetime(sub['SOE']).dt.to_period('Q').astype(str)
    sub['Year'] = pd.to_datetime(sub['SOE']).dt.to_period('Y').astype(str)
    
    # Plot bar chart for Pressure Ulcer Count by Type and sorted by month
    type_counts_by_month = pd.crosstab(sub['Month'], sub['Type']).fillna(0)
    type_counts_by_month = type_counts_by_month.div(type_counts_by_month.sum(axis=1), axis=0) * 100    
    type_counts_by_month = type_counts_by_month.sort_values('Month', ascending=True)
    
    custom_colors = ['#006837', '#b7e075', '#febe6f', '#a50026']  
    
    # Dropdown menu to choose time grouping
    time_grouping = st.selectbox('Choose the time grouping: ', ['Month', 'Quarter', 'Year'])

    # Resample the data based on the chosen time grouping
    if time_grouping == 'Quarter':
        type_counts_by_month = pd.crosstab(sub['Quarter'], sub['Type']).fillna(0)
        type_counts_by_month = type_counts_by_month.div(type_counts_by_month.sum(axis=1), axis=0) * 100    
        type_counts_by_month = type_counts_by_month.sort_values('Quarter', ascending=True)
        default_num_bars = min(len(type_counts_by_month), 12)
        num_bars = st.slider('Choose the timeframe for display: ', min_value=1, max_value=len(type_counts_by_month)-default_num_bars+2, value=1)
    elif time_grouping == 'Year':
        type_counts_by_month = pd.crosstab(sub['Year'], sub['Type']).fillna(0)
        type_counts_by_month = type_counts_by_month.div(type_counts_by_month.sum(axis=1), axis=0) * 100    
        type_counts_by_month = type_counts_by_month.sort_values('Year', ascending=True)
        default_num_bars = min(len(type_counts_by_month), 12)
        num_bars = st.slider('Choose the timeframe for display: ', min_value=1, max_value=len(type_counts_by_month)-default_num_bars+2, value=1)
    else:  # Default to 'Month'
        default_num_bars = min(len(type_counts_by_month), 12)
        num_bars = st.slider('Choose the timeframe for display: ', min_value=1, max_value=len(type_counts_by_month)-default_num_bars+2, value=1)
    
    # Check if there is data available for plotting
    if not type_counts_by_month.empty:
        # Plotting the chart with the selected window of bars
        plt.figure(figsize=(10, 6))
        ax = type_counts_by_month.iloc[num_bars:num_bars+default_num_bars].plot(kind='bar', title=f'Historical distribution of pressure ulcer by Type ({time_grouping})', stacked=True, color=custom_colors)
        
        plt.xlabel('Time')
        plt.ylabel('Percentage')
        
        # Move the legend to the right, outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Display the plot using Streamlit
        st.pyplot()
    else:
        st.warning("No data available for the selected time grouping.")
    
def plot_severity_counts(brad):
    
    unique_brad_patients = brad.sort_values('Visitdate', ascending=False).drop_duplicates('Name')
    type_counts = unique_brad_patients['Severity'].value_counts().sort_values(ascending=False)

    colors = plt.cm.RdYlGn(np.linspace(1, 0, len(type_counts)))
    
    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar', title='Severity levels count for all unique patients', color = colors)
    plt.xlabel('Severity')
    plt.ylabel('Count')

    # Add labels for each bar
    for i, count in enumerate(type_counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot()

def plot_severity_counts_by_month(brad):
    sub = brad.sort_values('Visitdate', ascending=False).copy()
    
    sub['Month'] = pd.to_datetime(sub['Visitdate']).dt.to_period('M').astype(str)
    sub['Quarter'] = pd.to_datetime(sub['Visitdate']).dt.to_period('Q').astype(str)
    sub['Year'] = pd.to_datetime(sub['Visitdate']).dt.to_period('Y').astype(str)
    
    # Plot bar chart for Pressure Ulcer Count by Type and sorted by month
    severity_counts_by_month = pd.crosstab(sub['Month'], sub['Severity']).fillna(0)
    severity_counts_by_month = severity_counts_by_month.div(severity_counts_by_month.sum(axis=1), axis=0) * 100    
    severity_counts_by_month = severity_counts_by_month.sort_values('Month', ascending=True)
    
    custom_colors = ['#006837', '#b7e075', '#febe6f', '#a50026']   
    
    # Dropdown menu to choose time grouping
    time_grouping = st.selectbox('Choose the time grouping:', ['Month', 'Quarter', 'Year'])

    # Resample the data based on the chosen time grouping
    if time_grouping == 'Quarter':
        severity_counts_by_month = pd.crosstab(sub['Quarter'], sub['Severity']).fillna(0)
        severity_counts_by_month = severity_counts_by_month.div(severity_counts_by_month.sum(axis=1), axis=0) * 100    
        severity_counts_by_month = severity_counts_by_month.sort_values('Quarter', ascending=True)
        default_num_bars = min(len(severity_counts_by_month), 12)
        num_bars = st.slider('Choose the timeframe for display: ', min_value=1, max_value=len(severity_counts_by_month)-default_num_bars+2, value=1)
    elif time_grouping == 'Year':
        severity_counts_by_month = pd.crosstab(sub['Year'], sub['Severity']).fillna(0)
        severity_counts_by_month = severity_counts_by_month.div(severity_counts_by_month.sum(axis=1), axis=0) * 100    
        severity_counts_by_month = severity_counts_by_month.sort_values('Year', ascending=True)
        default_num_bars = min(len(severity_counts_by_month), 12)
        num_bars = st.slider('Choose the timeframe for display: ', min_value=1, max_value=len(severity_counts_by_month)-default_num_bars+2, value=1)
    else:  # Default to 'Month'
        default_num_bars = min(len(severity_counts_by_month), 12)
        num_bars = st.slider('Choose the timeframe for display: ', min_value=1, max_value=len(severity_counts_by_month)-default_num_bars+2, value=1)

    # Check if there is data available for plotting
    if not severity_counts_by_month.empty:
        # Plotting the chart with the selected window of bars
        plt.figure(figsize=(10, 6))
        ax = severity_counts_by_month.iloc[num_bars:num_bars+default_num_bars].plot(kind='bar', title=f'Historical distribution of level of severity ({time_grouping})', stacked=True, color=custom_colors)
        
        plt.xlabel('Time')
        plt.ylabel('Percentage')
        
        # Move the legend to the right, outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Display the plot using Streamlit
        st.pyplot()
    else:
        st.warning("No data available for the selected time grouping.")


def braden_score_for_ulcer_patient_counts(ulcer_b):
    # Sort and count the Latest_Assessment_Score
    unique_ulcer_b_score = ulcer_b.sort_values('SOE', ascending=False).drop_duplicates('Name')
    score_counts = unique_ulcer_b_score['AssessmentAnswer'].value_counts().reset_index()

    min_value = score_counts.iloc[:, 0].min()
    max_value = score_counts.iloc[:, 0].max()
    colors = plt.cm.RdYlGn((score_counts.iloc[:, 0] - min_value) / (max_value - min_value))
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(score_counts.iloc[:, 0], score_counts.iloc[:, 1], color= colors)
    plt.xlabel('AssessmentAnswer')
    plt.ylabel('Number of Names')
    plt.title('Latest_Braden_Score_for_ulcer_patient_counts')

    # Add text labels above the bars
    for i, count in enumerate(score_counts.iloc[:, 1]):
        plt.text(score_counts.iloc[:, 0][i], count, str(count), ha='center', va='bottom', size=10, color='black')
        
    # Set x-axis ticks from the lowest to the highest AssessmentAnswer values with a step of 1
    plt.xticks(range(int(score_counts.iloc[:, 0].min()), int(score_counts.iloc[:, 0].max()) + 1))
    st.pyplot()

def location_counts(ulcer_b):
    name_location_count = ulcer_b.groupby('Name')['Location'].nunique().reset_index()
    name_location_count.columns = ['Name', 'Location_Count']
    
    location_count_name_counts = name_location_count.groupby('Location_Count')['Name'].count().reset_index()
    location_count_name_counts.columns = ['Location_Count', 'Name_Count']
    
    location_counts = location_count_name_counts['Location_Count']
    name_counts = location_count_name_counts['Name_Count']

    colors = plt.cm.RdYlGn(np.linspace(1, 0, len(location_counts)))

    plt.figure(figsize=(10, 6))
    plt.bar(location_counts, name_counts, color = colors)
    plt.xticks(range(1, max(location_counts) + 1))
    
    for i, count in enumerate(name_counts):
        plt.text(location_counts[i], count, str(count), ha='center', va='bottom')
    
    plt.title('Patients by Location Count')
    plt.xlabel('Ulcer location count')
    plt.ylabel('Patients count')
    st.pyplot()

def high_loc(ulcer_b):
    name_location_count = ulcer_b.groupby('Name')['Location'].nunique().reset_index()
    name_location_count.columns = ['Name', 'Location_Count']

    loc = name_location_count[name_location_count['Location_Count'] >=10]
    st.write("Patients with more than 10 ulcers:")
    st.write(loc)

def heal_rate_type(ulcer_b):    
    # Check if ulcer_b is not None
    if ulcer_b is None:
        return pd.DataFrame()  # Return an empty DataFrame or handle it based on your logic

    # Convert date columns to datetime objects for sorting
    ulcer_b['Onset'] = pd.to_datetime(ulcer_b['Onset'])
    ulcer_b['VisitDate'] = pd.to_datetime(ulcer_b['Visitdate'])
    ulcer_b['SOE'] = pd.to_datetime(ulcer_b['SOE'])
    
    # Check if sort_values is successful before assigning to ulcer_b
    try:
        ulcer_b = ulcer_b.sort_values(by=['Name', 'Location', 'SOE', 'VisitDate'], ascending=[True, True, True, False])
    except AttributeError as e:
        st.write(f"Error sorting DataFrame: {e}")
        return pd.DataFrame()  # Return an empty DataFrame or handle it based on your logic
    
    # Group by Name, Location, and SOE
    grouped = ulcer_b.groupby(['Name', 'Location', 'SOE'])
    
    # Initialize an empty list to store final results
    final_results = []
    
    # Iterate through groups and construct final results
    for (name, location, soe), group_df in grouped:
        onset_diff = (group_df['Onset'].max() - group_df['Onset'].min()).days
        visit_diff = (group_df['VisitDate'].max() - group_df['VisitDate'].min()).days
    
        if onset_diff < 60 and visit_diff < 60:
            types = group_df['Type'].tolist()
            assessment_scores = group_df['AssessmentAnswer'].tolist()
        else:
            latest_type = group_df.loc[group_df['VisitDate'].idxmax()]['Type']
            latest_score = group_df.loc[group_df['VisitDate'].idxmax()]['AssessmentAnswer']
            types = [latest_type]
            assessment_scores = [latest_score]  # Adding the latest assessment score
    
        latest_visit_date = group_df['VisitDate'].max()
        final_results.append({
            'Name': name,
            'Location': location,
            'SOE': soe,
            'types': types,
            'assessment_scores': assessment_scores,
            'latest_visit_date': latest_visit_date
        })
    
    # Create a DataFrame from final_results list
    df3 = pd.DataFrame(final_results)
    return df3

def heal_rate_braden_score(brad):    
    # Dictionary to store unique names as keys and their AssessmentAnswer, Visitdates, and woundID as values
    name_data = defaultdict(lambda: {"AssessmentAnswer": [], "Visitdates": [], "woundID": None})
    
    # Sort the merged dataframe by 'Name' and 'Visitdate' to ensure data is ordered correctly
    brad = brad.sort_values(by=['Name', 'Visitdate'])

    # Counter to generate unique wound IDs
    wound_id_counter = 1
    merged_df = brad.copy()
    
    # Process the data and group scores, dates, and assign wound IDs within 60-day windows
    for _, row in merged_df.iterrows():
        name = row["Name"]
        visit_date = row["Visitdate"]
        assessment_answer = row["AssessmentAnswer"]
    
        # Check if name already exists in the dictionary
        if name_data[name]["Visitdates"]:
            # Check if there are previous visit dates
            last_visit_date = name_data[name]["Visitdates"][0]
            # Check if the visit date difference is less than or equal to 60 days
            if (visit_date - last_visit_date).days <= 60:
                name_data[name]["AssessmentAnswer"].append(assessment_answer)
                name_data[name]["Visitdates"].append(visit_date)
            else:
                # If the visit date difference is more than 60 days, create a new entry
                name_data[name]["AssessmentAnswer"] = [assessment_answer]
                name_data[name]["Visitdates"] = [visit_date]
                # Assign a unique wound ID for the new entry
                name_data[name]["woundID"] = wound_id_counter
                wound_id_counter += 1
        else:
            # If there are no previous visit dates, add the current date and score
            name_data[name]["AssessmentAnswer"].append(assessment_answer)
            name_data[name]["Visitdates"].append(visit_date)
            # Assign a unique wound ID for the new entry
            name_data[name]["woundID"] = wound_id_counter
            wound_id_counter += 1
    
    # Adding new columns to the merged dataframe
    merged_df["Sorted_AssessmentAnswer"] = merged_df["Name"].apply(lambda x: name_data[x]["AssessmentAnswer"])
    merged_df["Sorted_Visitdates"] = merged_df["Name"].apply(lambda x: name_data[x]["Visitdates"])
    merged_df["woundID"] = merged_df["Name"].apply(lambda x: name_data[x]["woundID"])
    
    # Dropping the original AssessmentAnswer column
    merged_df.drop(columns=['AssessmentAnswer'], inplace=True)
    merged_df.drop_duplicates(subset=['woundID'], keep='first', inplace=True)
    return merged_df

def heal_rate_merge(merged_df,df3): 
    # Merge based on 'Name' and conditions for 'SOE' and 'Visitdate'
    result = pd.merge(df3, merged_df[['Name', 'Sorted_AssessmentAnswer', 'Visitdate', 'Sorted_Visitdates', 'Age_as_of_visit']], how='left', on='Name')
    # Filter rows where Visitdate is >= SOE and not greater than 60 days
    result = result[(result['Visitdate'] >= result['SOE']) & (result['Visitdate'] - result['SOE'] <= pd.Timedelta(days=60))]
    # Reset index if needed
    result.reset_index(drop=True, inplace=True)
    return result

def heal_logic(result):
    for index, row in result.iterrows():
        assessment_scores_row = row["Sorted_AssessmentAnswer"]
        types = row["types"]
        types = [types] if isinstance(types, int) else types
        if types is None:
            # Handle the case when types is None (customize based on your logic)
            categorization = "Unknown"
        else:    
            # Check if assessment_scores list is non-empty
            if len(assessment_scores_row) > 0:
                # Check the conditions and categorize the data
                if len(assessment_scores_row) == 1:
                    if assessment_scores_row[0] >= 19:
                        # check stage
                        if len(types) ==1 and types[0] ==4:
                            categorization = "Pending"
                        elif len(types) ==1 and types[0] ==3:
                            categorization = "Pending"
                        else:
                            categorization = "Healed"
                    else:
                        if len(types) == 1 and types[0] == 4:
                            categorization = "Pending"
                        elif len(types) == 1 and types[0] == 3:
                            categorization = "Pending"
                        elif len(types) >= 2:
                            # stage decrease
                            if types[0] > types[-1]:
                                categorization = "Healing"
                            elif types[0] > types[-1] and types[-1] == 4:
                                categorization = "Healing"
                            elif types[0] > types[-1] and types[-1] == 3:
                                categorization = "Healing"
                            # stage increase
                            elif types[0] < types[-1]:
                                categorization = "Worse"
                            # stage same
                            else:
                                categorization = "Healing"
                        else:
                            categorization = "Pending"
                elif len(assessment_scores_row) >= 2:
                        # check stages
                    if len(types) >= 2:
                        if types[0] > types[-1]:
                            categorization = "Healing"
                        elif types[0] > types[-1] and types[-1] == 4:
                            categorization = "Healing"
                        elif types[0] > types[-1] and types[-1] == 3:
                            categorization = "Healing"
                        elif types[0] < types[-1]:
                            categorization = "Worse"
                        # stage stay same
                        else:
                            categorization = "Healing"
                    # only one record
                    else:
                        categorization = "Healed"
                else:
                    categorization = "Pending"
            else:
                categorization = "Pending"
        
            # Adding the categorization to the data DataFrame for the current row
            result.at[index, "Categorization"] = categorization

    # Dropping the original AssessmentAnswer column
    # result.drop(columns=['assessment_scores'], inplace=True)
    result['last_assessment_score'] = result['Sorted_AssessmentAnswer'].apply(lambda x: x[-1])
    return result


def Cate_given_brad(result):
    # Define your custom color set with four colors
    # custom_colors = ['#1f77b4',  '#2ca02c', '#ff7f0e','#d62728']
    custom_colors = ['#006837', '#b7e075', '#febe6f', '#a50026']  
    
    # Group by the 'last_assessment_score' column and 'Categorization', then count the occurrences
    agg_result = result.groupby(['last_assessment_score', 'Categorization']).size().reset_index(name='Count')
    
    # Pivot the result DataFrame to prepare it for plotting
    pivot_result = agg_result.pivot(index='last_assessment_score', columns='Categorization', values='Count')
    
    # Create a bar plot with custom colors
    ax = pivot_result.plot(kind='bar', stacked=True, color=custom_colors)
    
    # Rest of your code remains the same
    plt.xlabel('Last Assessment Score')
    plt.ylabel('Count')
    plt.title('Count of Categorization by Last Assessment Score')
    plt.legend(title='Categorization', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # Annotate bars with the count
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{int(height)}', (x + width / 2, y + height / 2), ha='center', fontsize = 8)
    
    st.pyplot()
    
def Cate_given_brad_perc(result):    

    # custom_colors = ['#1f77b4',  '#2ca02c', '#ff7f0e','#d62728']
    custom_colors = ['#006837', '#b7e075', '#febe6f', '#a50026']  
    
    # Group by the 'last_assessment_score' column and 'Categorization', then calculate the sum of counts
    agg_result = result.groupby(['last_assessment_score', 'Categorization']).size().reset_index(name='Count')
    
    # Calculate total counts for each 'last_assessment_score'
    total_counts = agg_result.groupby('last_assessment_score')['Count'].transform('sum')
    
    # Calculate percentages for each label in each group
    agg_result['Percentage'] = (agg_result['Count'] / total_counts) * 100
    
    # Pivot the agg_result DataFrame to prepare it for plotting
    pivot_result = agg_result.pivot(index='last_assessment_score', columns='Categorization', values='Percentage')
    
    # Create a stacked bar plot for percentages
    ax = pivot_result.plot(kind='bar', stacked=True, color=custom_colors)
    plt.xlabel('Last Assessment Score')
    plt.ylabel('Percentage')
    plt.title('Percentage of Categorization by Last Assessment Score')
    plt.legend(title='Categorization', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # Annotate bars with percentages only
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        if height != 0:
            percentage_value = f'{height:.1f}%'
            ax.annotate(percentage_value, (x + width / 2, y + height / 2), ha='center', fontsize = 8)
    
    st.pyplot()

def calculate_healing_speed(row):
    assessment_scores = row['Sorted_AssessmentAnswer']
    visit_dates = row['Sorted_Visitdates']

    # Check if the categorization is 'healing' or 'healed'
    if len(assessment_scores) >= 2 and len(visit_dates) >= 2:
        first_score = assessment_scores[0]
        last_score = assessment_scores[-1]
        first_date = pd.to_datetime(visit_dates[0])
        last_date = pd.to_datetime(visit_dates[-1])

        # Check if the dates are different before dividing
        if (last_date - first_date).days != 0:
            # Calculate the healing speed in units per day
            if last_score >= 19 and first_score < 19:
                healing_speed = (last_score - first_score) / (last_date - first_date).days
                return healing_speed
    # Return None if conditions are not met
    return None

def heal_speed_by_age(merged_df):
    
    # Apply the function row-wise
    merged_df['HealingSpeed'] = merged_df.apply(calculate_healing_speed, axis=1)
    merged_df = merged_df.dropna(subset=['HealingSpeed'])
    # Remove outliers where healing speed is larger than the threshold
    merged_df = merged_df[merged_df['HealingSpeed'] <=  2]
    
    age_bins = np.arange(0, 106, 5)
    
    # Create age groups and calculate average healing speed for each group
    merged_df['AgeGroup'] = pd.cut(merged_df['Age_as_of_visit'], bins=age_bins, right=False)
    age_grouped = merged_df.groupby('AgeGroup')['HealingSpeed'].mean()
    
    # Plotting the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(age_grouped.index.astype(str), age_grouped.values, color='skyblue')
    plt.xlabel('Age Group')
    plt.ylabel('Average Healing Speed')
    plt.title('Average Healing Speed by Age Group (5-year intervals)')
    plt.xticks(rotation=45)
    st.pyplot()
 
def find_worse(result):
    worse_result = result[result['Categorization'] == 'Worse']
    st.write('Patients whose condition got worse: ')
    st.write(worse_result)
    
def got_ulcer(brad,ulcer_b):
    brad['got_ulcer'] = brad["Name"].apply(lambda x: x in list(ulcer_b["Name"]))
    brad[brad['got_ulcer']== True].sort_values('Name',ascending = True)
    return brad

def vulnerable(brad):
    brad = brad.sort_values('Visitdate', ascending = False)
    sub = brad.drop_duplicates(subset=['Name'], keep='first')
    sub = sub.drop(['Textbox65', 'AssessmentQuestion'],axis=1)
    vul = sub[(sub['Severity'] == "High Risk") & (sub['got_ulcer'] == False)]
    st.write('Patients who are vulnerable and might get pressure ulcers in the future: ')
    st.write(vul)

def duration(brad):
    dura = brad.groupby(['Name', 'Visitdate']).agg(list).reset_index()
    
    # Sort the DataFrame by 'Visitdate' within each 'Name' group
    dura.sort_values(by=['Name', 'Visitdate'], inplace=True)
    
    # Group by 'Name' and select the first and last visit dates
    dura['first_visit'] = dura.groupby('Name')['Visitdate'].transform('first')
    dura['last_visit'] = dura.groupby('Name')['Visitdate'].transform('last')
    dura['duration'] = (dura['last_visit'] - dura['first_visit']).dt.days
    
    # Reset the index to have a clean DataFrame
    dura.reset_index(inplace=True)
    # Merge 'brad_b' with 'brad_v' on the 'Name' column
    brad = pd.merge(brad, dura[['Name', 'duration']].drop_duplicates(), on='Name', how='left')
    brad = brad.dropna()
    return brad


# predictive modeling --------------------------------------------------------------------------------------#

def find_categorical_columns(data):
    # Identify columns with object data type
    categorical_columns = [col for col, dtype in data.dtypes.items() if pd.api.types.is_object_dtype(dtype)]
    return categorical_columns

def convert_to_factors(data, categorical_columns):
    for col in categorical_columns:
        data[col] = data[col].astype('category').cat.codes
    return data

def SVM(brad):
    seed = 1031

    # Select features and target variable
    #features = ["AssessmentAnswer", "Age_as_of_visit", "duration"]
    features = ["AssessmentAnswer", "ServiceCode", "Visitdate", "Worker_type", "Age_as_of_visit", "duration"]
    target = 'got_ulcer'

    # Create a subset of brad with only selected columns
    sub_brad = brad[features + [target]]
    sub_brad = convert_to_factors(sub_brad,["ServiceCode", "Worker_type"])
    sub_brad["Visitdate"] = sub_brad["Visitdate"].dt.month.astype(int)

    # Convert the 'got_ulcer' column to binary labels
    label_encoder = LabelEncoder()
    sub_brad.loc[:, 'got_ulcer'] = sub_brad['got_ulcer'].astype(int)

    sub_brad = sub_brad.sort_values(by='Visitdate', ascending=False)

    # Use the most recent 10% of data as the test set
    split_index = int(0.9 * len(sub_brad))
    train_data = sub_brad.iloc[:split_index, :]
    test_data = sub_brad.iloc[split_index:, :]

    # Output timeframe of the test set
    min_test_date = brad.loc[brad.index[split_index], 'Visitdate']
    max_test_date = brad.loc[brad.index[-1], 'Visitdate']
    st.write(f"Timeframe of Test Set: from {min_test_date} to {max_test_date}")

    # Scale only numeric features
    scaler = StandardScaler()
    train_data.iloc[:, :-1] = scaler.fit_transform(train_data.iloc[:, :-1])  # Scale all columns except the last one
    test_data.iloc[:, :-1] = scaler.fit_transform(test_data.iloc[:, :-1])
    
    X_train, y_train = train_data.iloc[:, :-1], train_data[target]
    X_test, y_test = test_data.iloc[:, :-1], test_data[target]

    # Estimate time to train SVM model
    dataset_size = len(X_train)
    complexity_factor = 0.0001  # Adjust this based on your model and dataset complexity

    estimated_time = round(dataset_size * complexity_factor*5, 2)
    st.write(f"Estimated time to train SVM model: {estimated_time} seconds")

    start_time = time.time()

   # Train the SVM model
    svm_model = SVC(kernel='rbf', C=1, gamma=0.1, max_iter=20000, random_state=seed)
    svm_model.fit(X_train, y_train)

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    st.write(f"Actual time taken to train SVM model: {elapsed_time} seconds")

    Inv_X_test = pd.DataFrame(scaler.inverse_transform(X_test), columns=X_test.columns)
    
    # Make predictions on the test set
    svm_pred = svm_model.predict(X_test)
    
    # Calculate accuracy
    svm_accuracy = round(accuracy_score(y_test, svm_pred),4)
    st.write(f"SVM Accuracy: {svm_accuracy}")
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, svm_pred)
    
    # Display confusion matrix with labels
    conf_matrix_display = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    st.write("Confusion Matrix:")
    st.write(conf_matrix_display)
    
    # Calculate AUC
    svm_auc_value = round(roc_auc_score(y_test, svm_pred),4)
    st.write(f"SVM AUC: {svm_auc_value}")
    
    # Plot decision boundary
    svm_plot_data = pd.concat([X_test, y_test], axis=1)
    svm_plot_data['svm_pred'] = svm_model.predict(X_test)
    
    # Create a column indicating correct or incorrect predictions
    svm_plot_data['prediction_correct'] = svm_plot_data['got_ulcer'] == svm_plot_data['svm_pred']
    
    # Plot SVM decision boundary with jitter using Plotly
    fig = px.scatter(Inv_X_test.join(svm_plot_data[['got_ulcer', 'prediction_correct']]), 
                     x='AssessmentAnswer', y='Age_as_of_visit', color='prediction_correct',
                     symbol='got_ulcer', opacity=0.7, size_max=10,
                     color_discrete_map={True: 'green', False: 'red'},
                     symbol_map={0: 'circle', 1: 'square'})

    fig.update_xaxes(range=[Inv_X_test['AssessmentAnswer'].min(), Inv_X_test['AssessmentAnswer'].max()],
                 tickmode='linear', dtick=1)
    
    fig.update_layout(title_text="SVM Decision Boundary",
                      xaxis_title="AssessmentAnswer",
                      yaxis_title="Age_as_of_visit",
                      legend_title="Prediction Correctness")
    
    # Show the figure
    st.plotly_chart(fig)

# ----------------------------------------------------------------------------------------------------------------#

def merge_with_birth(brad, birth):

    brad = brad.merge(birth, left_on='Name', right_on='Name')
    brad = brad.dropna()
    brad['Age_as_of_visit'] = (brad['Visitdate'].dt.year - brad['DOB']).astype(int)
    return brad

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

    return ulcer_b  

def create_table_of_contents(subheaders):
    st.sidebar.header("Table of Contents")
    for subheader in subheaders:
        st.sidebar.markdown(f"- [{subheader}](#{subheader.lower().replace(' ', '-')} )")

def convert_df_to_csv_bytes(df):
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    return csv_bytes

def create_label(dataset_name, total_width=60):
    prefix = "Download "
    suffix = " dataset"
    # Calculate the remaining width for the dataset name
    remaining_width = total_width - len(prefix) - len(suffix)
    # Center the dataset name within the remaining width
    centered_name = dataset_name.center(remaining_width)
    return f"{prefix}{centered_name}{suffix}"
    
# ----------------------------------------------------------------------------------------------------------------#

image_path = "nascentia_logo.png"
st.image(image_path, width = 100)
st.title("Nascentia Pressure Ulcer Data Analyzer")

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        entered_username = st.session_state["username"]
        entered_password = st.session_state["password"]
    
        # Check if the entered username exists in the secrets
        if entered_username in st.secrets["users"]:
            # Compare the entered password with the stored one
            correct_password = st.secrets["users"][entered_username]
            if hmac.compare_digest(entered_password, correct_password):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store the username or password.
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

if not check_password():
    st.stop()

def logout():
    # Reset or remove login-related session state variables
    for key in ['password_correct', 'username', 'password']:
        if key in st.session_state:
            del st.session_state[key]

if 'password_correct' in st.session_state and st.session_state['password_correct']:
    def main():
    
        subheaders = ["Instructions", "Data Upload", "Data Outlook", "Filter Data by Dates", "Patient Search", "Severity Overview", "Ulcer Type Overview", "Heal Rate Analysis", "Machine Learning", "Patients Spotlight", "Processed Data Download", "Pressure Ulcer Image Classifier"]
        create_table_of_contents(subheaders)
        
        st.subheader("Instructions:")
        st.write("**1. Pull the equivalent data from the database for the past 3 years from the day intended for analysis.**")
        st.write("**2. Save the data as \".csv\" format and upload them in the CORRECT-ORDER.**")
        st.write("**3. Wait for the system to process and plot the analysis results.**")
        st.write("**4. Try using the sidebar table of content to jump to desire sections.**")
        st.write("**5. Try manipulating plots with interactive tables and widgets.**")
        st.write("**Note: The machine learning model will be trained with historical data and predict the results of the most recent patients. Those who are predicted to get an ulcer but actually do not may be interpreted by the model as being vulnerable to getting an ulcer in the future.**")
    
        st.subheader("Data Upload")
        birth = upload_birth_csv()
        ulcer = upload_ulcer_csv()
        brad = upload_brad_csv()
    
        if birth is not None:
            st.subheader("Data Outlook")
            process_birth_data(birth)
            st.write(f"Length of 'Birthday Data': {len(birth)}")
        
        if ulcer is not None:
            process_ulcer_data(ulcer)
            st.write(f"Length of 'Pressure Ulcer Data': {len(ulcer)}")
    
        # Display the processed brad dataset
        if brad is not None:
            process_brad_data(brad)
            brad = duration(brad)
            st.write(f"Length of 'Physical Assessment Data': {len(brad)}")
    
        if ulcer is not None and brad is not None:
            ulcer, brad = filter_date(ulcer, brad)
            brad = merge_with_birth(brad, birth)
    
        if ulcer is not None and brad is not None:
            brad = got_ulcer(brad,ulcer)
            ulcer_b = merge_and_process_data(ulcer, brad)  # Get the processed DataFrame
            st.write(f"Length of 'Pressure Ulcer Data merge Physical Assessment Data': {len(ulcer_b)}")
            st.write("Preview of 'Pressure Ulcer Data merge Physical Assessment Data' DataFrame:")
            st.write(ulcer_b)
    
            # Allow user to input a patient ID
            st.subheader("Patient Search")
            patient_id = st.text_input("**Enter Patient ID (in format of First-Last, e.g. 12-345) for their Braden score history:**")
            
            # Check if the patient ID is provided
            if patient_id:
                # Plot line chart for the specified patient
                plot_patient_data(patient_id, brad, ulcer)
    
        if brad is not None:
            st.subheader("Severity Overview")
            plot_severity_counts(brad)
            plot_severity_counts_by_month(brad)
            
        if ulcer is not None and brad is not None:
            st.subheader("Ulcer Type Overview")
            plot_ulcer_counts(ulcer_b)
            plot_ulcer_counts_by_month(ulcer_b)
            braden_score_for_ulcer_patient_counts(ulcer_b)
            location_counts(ulcer_b)
            
            df3 = heal_rate_type(ulcer_b)
            merged_df = heal_rate_braden_score(brad)
            result = heal_rate_merge(merged_df,df3)
            result = heal_logic(result)
    
            st.subheader("Heal Rate Analysis")
            Cate_given_brad(result)
            Cate_given_brad_perc(result)
            heal_speed_by_age(merged_df)
    
            st.subheader("Machine Learning")
            SVM(brad)
             
            st.subheader("Patients Spotlight")
            high_loc(ulcer_b)
            find_worse(result)
            vulnerable(brad)

            st.subheader("Processed Data Download")
            
            
            st.download_button(
                                    label="**Download-------------------'birthday'----------------------dataset**",
                                    data= convert_df_to_csv_bytes(birth),
                                    file_name='birthday_processed.csv',
                                    mime='text/csv',
                                    use_container_width = False
                                )
            st.download_button(
                                    label="**Download----------------'ulcer / pchart'-------------------dataset**",
                                    data= convert_df_to_csv_bytes(ulcer),
                                    file_name='ulcer_processed.csv',
                                    mime='text/csv',
                                    use_container_width = False
                                )
            st.download_button(
                                    label="**Download---'braden score / physical assessment'---dataset**",
                                    data= convert_df_to_csv_bytes(brad),
                                    file_name='brad_processed.csv',
                                    mime='text/csv',
                                    use_container_width = False
                                )
            st.download_button(
                                    label="**Download--------------------'merged'-----------------------dataset**",
                                    data= convert_df_to_csv_bytes(ulcer_b),
                                    file_name='ulcer_b.csv',
                                    mime='text/csv',
                                    use_container_width = False
                                )
            st.download_button(
                                    label="**Download---------------'healing analysis'-----------------dataset**",
                                    data= convert_df_to_csv_bytes(result),
                                    file_name='healing.csv',
                                    mime='text/csv',
                                    use_container_width = False
                                )   
        st.subheader("Pressure Ulcer Image Classifier")
        
        MODEL_PATH = "last.pt"
        model_dict = torch.load(MODEL_PATH)
        model = model_dict['model']
        model.eval()
        
        option = st.radio("**Choose your input method:**", ('Upload an Image', 'Take a Photo'))
        
        uploaded_file = None
        
        if option == 'Upload an Image':
            uploaded_file = st.file_uploader("**Choose an image...**", type=["jpg", "jpeg", "png"])
        elif option == 'Take a Photo':
            camera_input = st.camera_input(label="**Take a photo...**", label_visibility="collapsed")
            if camera_input is not None:
                # Read the image file buffer as a PIL Image
                img = Image.open(camera_input)
        
                # Convert PIL Image to a BytesIO object in JPEG format
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                buffer.seek(0)
                uploaded_file = buffer
        
        if uploaded_file is not None:
            detections = load_and_infer_image(uploaded_file, model)
            predicted_class_label, detection_result = display_results(detections)
            with st.expander("**Click to view uploaded image**"):
                if isinstance(uploaded_file, io.BytesIO):
                    uploaded_image = Image.open(uploaded_file)
                    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
                else:
                    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write(f"**Predicted Skin Condition: {predicted_class_label}**")
            st.write(f"**Detection Confidence: {detection_result}**")
        
        st.markdown("**Appendix: [The logic of graphs and analysis for reference]**"
                    "(https://drive.google.com/file/d/1fdlZvz1MJB2MUytRCtJgErGbnS_SCLqY/view?usp=sharing)")
            
    if __name__ == "__main__":
        main()

