#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import sys
import ast
from datetime import datetime as dt, timedelta
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict

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

    print("After modification - Unique values in 'Type':", ulcer['Type'].unique())

    return ulcer  # Return the modified DataFrame

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

    colors = plt.cm.RdYlGn(np.linspace(1, 0, len(type_counts)))
    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar', title='Pressure Ulcer Count by Type', color = colors)
    plt.xlabel('Type')
    plt.ylabel('Count')

    # Add labels for each bar
    for i, count in enumerate(type_counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot()


def plot_severity_counts(brad):
    
    unique_brad_patients = brad.sort_values('Visitdate', ascending=False).drop_duplicates('Name')
    type_counts = unique_brad_patients['Severity'].value_counts().sort_values(ascending=False)

    colors = plt.cm.RdYlGn(np.linspace(1, 0, len(type_counts)))
    
    plt.figure(figsize=(10, 6))
    type_counts.plot(kind='bar', title='Severity levels count over all', color = colors)
    plt.xlabel('Severity')
    plt.ylabel('Count')

    # Add labels for each bar
    for i, count in enumerate(type_counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot()

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

    
def heal_rate_braden_score(brad,ulcer):    
    # Dictionary to store unique names as keys and their AssessmentAnswers, Visitdates, and woundID as values
    name_data = defaultdict(lambda: {"AssessmentAnswers": [], "Visitdates": [], "woundID": None})
    
    # Sort the merged dataframe by 'Name' and 'Visitdate' to ensure data is ordered correctly
    brad = brad.sort_values(by=['Name', 'Visitdate'])
    # ulcer_b['SOE'] = pd.to_datetime(ulcer_b['SOE'])
    # ulcer_b['Visitdate'] = pd.to_datetime(ulcer_b['Visitdate'])

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
                name_data[name]["AssessmentAnswers"].append(assessment_answer)
                name_data[name]["Visitdates"].append(visit_date)
            else:
                # If the visit date difference is more than 60 days, create a new entry
                name_data[name]["AssessmentAnswers"] = [assessment_answer]
                name_data[name]["Visitdates"] = [visit_date]
                # Assign a unique wound ID for the new entry
                name_data[name]["woundID"] = wound_id_counter
                wound_id_counter += 1
        else:
            # If there are no previous visit dates, add the current date and score
            name_data[name]["AssessmentAnswers"].append(assessment_answer)
            name_data[name]["Visitdates"].append(visit_date)
            # Assign a unique wound ID for the new entry
            name_data[name]["woundID"] = wound_id_counter
            wound_id_counter += 1
    
    # Adding new columns to the merged dataframe
    merged_df["Sorted_AssessmentAnswers"] = merged_df["Name"].apply(lambda x: name_data[x]["AssessmentAnswers"])
    merged_df["Sorted_Visitdates"] = merged_df["Name"].apply(lambda x: name_data[x]["Visitdates"])
    merged_df["woundID"] = merged_df["Name"].apply(lambda x: name_data[x]["woundID"])
    
    # Dropping the original AssessmentAnswers column
    merged_df.drop(columns=['AssessmentAnswer'], inplace=True)
    merged_df.drop_duplicates(subset=['woundID'], keep='first', inplace=True)

    #merged_df['Visitdate'] = pd.to_datetime(merged_df['Visitdate'])
    
    # Merge based on 'Name' and conditions for 'SOE' and 'Visitdate'
    merged_df2 = pd.merge(ulcer, merged_df, how='inner', on='Name')
    # Filter rows where Visitdate is >= SOE and not greater than 60 days
    merged_df2 = merged_df2[(merged_df2['Visitdate'] >= merged_df2['SOE']) & (merged_df2['Visitdate'] - merged_df2['SOE'] <= pd.Timedelta(days=60))]
    
    # Reset index if needed
    merged_df2.reset_index(drop=True, inplace=True)
    
    result = merged_df2
    st.write(result)
    return result
    
def heal_logic(result):
    for index, row in result.iterrows():
        assessment_scores_row = row["Sorted_AssessmentAnswers"]
        types_str = row["Type"]

        # Convert string representation of list to actual list
        types = ast.literal_eval(types_str)
    
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
                    if len(types) >= 2:
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

    # Dropping the original AssessmentAnswers column
    result.drop(columns=['assessment_scores'], inplace=True)
    result['last_assessment_score'] = result['Sorted_AssessmentAnswers'].apply(lambda x: x[-1])
    
    return result

# #------------------------------------------------------------------------------------------------------
# def Dist_Cate_Labels(result):
#     # Count the occurrences of each category
#     category_counts = result['Categorization'].value_counts()
    
#     # Create a bar plot with counts on top of the bars
#     plt.figure(figsize=(8, 6))
#     ax = category_counts.plot(kind='bar', color='skyblue')
#     plt.title('Count Distribution of Categorization Labels')
#     plt.xlabel('Categorization Labels')
#     plt.ylabel('Count')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     # Add counts on top of the bars
#     for p in ax.patches:
#         ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
#                     ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
#     st.pyplot()
    
    # # Define your custom color set with four colors
    # custom_colors = ['#1f77b4',  '#2ca02c', '#ff7f0e','#d62728']
    
    # # Group by the 'last_assessment_score' column and 'Categorization', then count the occurrences
    # result = final_logic_dataset.groupby(['last_assessment_score', 'Categorization']).size().reset_index(name='Count')
    
    # # Pivot the result DataFrame to prepare it for plotting
    # pivot_result = result.pivot(index='last_assessment_score', columns='Categorization', values='Count')
    
    # # Create a bar plot with custom colors
    # ax = pivot_result.plot(kind='bar', stacked=True, color=custom_colors)
    
    # # Rest of your code remains the same
    # plt.xlabel('Last Assessment Score')
    # plt.ylabel('Count')
    # plt.title('Count of Categorization by Last Assessment Score')
    # plt.legend(title='Categorization')
    # plt.xticks(rotation=90)
    
    # # Annotate bars with the count
    # for p in ax.patches:
    #     width = p.get_width()
    #     height = p.get_height()
    #     x, y = p.get_xy()
    #     ax.annotate(f'{int(height)}', (x + width / 2, y + height / 2), ha='center')
    
    # plt.show()
    
    
    # import matplotlib.pyplot as plt
    
    # # Group by the 'last_assessment_score' column and 'Categorization', then calculate the sum of counts
    # result = final_logic_dataset.groupby(['last_assessment_score', 'Categorization']).size().reset_index(name='Count')
    
    # # Calculate total counts for each 'last_assessment_score'
    # total_counts = result.groupby('last_assessment_score')['Count'].transform('sum')
    
    # # Calculate percentages for each label in each group
    # result['Percentage'] = (result['Count'] / total_counts) * 100
    
    # # Pivot the result DataFrame to prepare it for plotting
    # pivot_result = result.pivot(index='last_assessment_score', columns='Categorization', values='Percentage')
    
    # # Create a stacked bar plot for percentages
    # ax = pivot_result.plot(kind='bar', stacked=True,color=custom_colors)
    # plt.xlabel('Last Assessment Score')
    # plt.ylabel('Percentage')
    # plt.title('Percentage of Categorization by Last Assessment Score')
    # plt.legend(title='Categorization')
    # plt.xticks(rotation=45)
    
    # # Annotate bars with percentages only
    # for p in ax.patches:
    #     width = p.get_width()
    #     height = p.get_height()
    #     x, y = p.get_xy()
    #     if height != 0:
    #         percentage_value = f'{height:.1f}%'
    #         ax.annotate(percentage_value, (x + width / 2, y + height / 2), ha='center')
    
    # st.pyplot()
    
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
    st.title("Nascentia Pressure Ulcer Data Analyzer")

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
        braden_score_for_ulcer_patient_counts(ulcer_b)
        location_counts(ulcer_b)
        heal_rate_braden_score(brad,ulcer)
        #result = heal_logic(result)
        # Dist_Cate_Labels(result)
    st.markdown("Appendix: [The logic of graphs and analysis for reference]"
            "(https://drive.google.com/file/d/1gyZnA_mfkNlwyOyjKlLGgIH7LiEUQvZQ/view?usp=share_link)")


        
if __name__ == "__main__":
    main()

