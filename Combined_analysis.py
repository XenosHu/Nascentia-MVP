#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import re
import sys
from datetime import datetime as dt, timedelta
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error, r2_score


#!pip install ipywidgets
from ipywidgets import interact
#!pip install streamlit
#pip install --upgrade arrow
import streamlit as st


# In[ ]:


st.title('Nascentia Analyze App')


# In[2]:


file_path = 'C:\\Users\\huxia\\Downloads\\capstone\\dataset\\'
birth_path = file_path + 'birth.xlsx'
ulcer_path = file_path + 'ulcer.xlsx'
brad_path = file_path + 'brad.xlsx'
case_path = file_path + 'case.xlsx'


# In[3]:


case_column_types = {
    'Case Manager': str,
    'Name': str,
    'IsPrimary': str,
    'Client Status': str,
    'Start_of_Episode': 'datetime64[ns]',
    'Episode_Status': str,
    'lblTotalAssignmentsByCaseManager': str,
    'TotalAssignmentsByCaseManager': 'int64',
    'case_manager_name': str,
    'case_manager_role': str,
    'case_manager_type': str
}

birth_column_types = {
    'Name': str,
    'DOB': 'int64'
}

ulcer_column_types = {
    'Name': str,
    'SOE': 'datetime64[ns]',
    'Location': str,
    'Type': 'int64',
    'Onset': 'datetime64[ns]',
    'Activated': 'datetime64[ns]'
}

brad_column_types = {
    'Textbox65': str,
    'Visitdate': 'datetime64[ns]',
    'AssessmentQuestion': str,
    'AssessmentAnswer': 'int64',
    'ServiceCode': str,
    'Name': str,
    'Severity': str
}

# Read the Excel files with specified column types
case = pd.read_excel(case_path, index_col=None, dtype=case_column_types)
birth = pd.read_excel(birth_path, index_col=None, dtype=birth_column_types)
ulcer = pd.read_excel(ulcer_path, index_col=None, dtype=ulcer_column_types)
brad = pd.read_excel(brad_path, index_col=None, dtype=brad_column_types)


# # HCHB_ClientBirthday (birth)

# In[4]:


# birth['DOB'] = birth['DOB'].dt.year
# birth['Name'] = birth['Name'].str.split('Last').str[1].str.split(', First').str.join('-')
# birth = birth[birth['DOB'].notna()]
# birth = birth.sort_values('DOB', ascending = True)
# birth


# In[5]:


# dup_birth = birth[birth['Name'].duplicated(keep=False)]
# dup_birth = dup_birth.sort_values('Name', ascending = True)

# dup_birth['is_in_case'] = dup_birth["Name"].apply(lambda x: x in list(case["Name"]))
# dup_ncase = list(dup_birth[dup_birth['is_in_case']== True]['Name'])
# dup_birth['is_in_case'] = dup_birth["Name"].apply(lambda x: x in list(brad["Name"]))
# dup_nbrad = list(dup_birth[dup_birth['is_in_case']== True]['Name'])
# dup_birth['is_in_case'] = dup_birth["Name"].apply(lambda x: x in list(ulcer["Name"]))
# dup_nulcer = list(dup_birth[dup_birth['is_in_case']== True]['Name'])

# case = case[~case['Name'].isin(dup_ncase)]
# brad = brad[~brad['Name'].isin(dup_nbrad)]
# ulcer = ulcer[~ulcer['Name'].isin(dup_nulcer)]
# birth = birth[~birth['Name'].isin(dup_birth["Name"])]


# In[6]:


# res = birth.copy()
# res['is_in_case'] = res["Name"].apply(lambda x: x in list(case["Name"]))
# res['is_in_brad'] = res["Name"].apply(lambda x: x in list(brad["Name"]))
# res['is_in_ulcer'] = res["Name"].apply(lambda x: x in list(ulcer["Name"]))


# In[7]:


# #Out of 36911 valid patients in birthday dataset
# print(len(birth))
# print(len(res[res['is_in_case']== True]))
# print(len(res[res['is_in_brad']== True]))
# print(len(res[res['is_in_ulcer']== True]))


# # HCHB_CaseManegerAssignment (case)

# In[8]:


episode_status_counts = case['Episode_Status'].value_counts()
total_count = len(case)

# Create a DataFrame to store the results
epis = pd.DataFrame({
    'Episode_Status': episode_status_counts.index,
    'Count': episode_status_counts.values,
    'Percentage %': (episode_status_counts / total_count * 100).round(2)
})

#epis


# # JR-RW_PChart (ulcer)

# # HCHB_Physical.Assessment.Report.Braden (brad)

# In[9]:


brad.sort_values(by='Visitdate', ascending=True)
#brad['AssessmentAnswer'] = ulcer["AssessmentAnswer"].astype(int)
brad['Worker_name'] = brad['Textbox65'].str.split(':').str[1].str.split(',').str[0]
brad['Worker_type'] = brad['Textbox65'].str.rsplit(',').str[1]


# In[10]:


brad_sub = brad.drop_duplicates(subset='Worker_name', keep='first')
brad_sub.reset_index(drop=True, inplace=True)

role_counts = brad_sub['Worker_type'].value_counts().sort_values(ascending=True)
role_counts.plot(kind='barh', title='Count of worker roles')
plt.xlabel('Roles')
plt.ylabel('Count')
plt.show()


# In[11]:


res_b = brad[["Name", "AssessmentAnswer","Worker_type"]]
res_b['is_in_ulcer'] = res_b["Name"].apply(lambda x: x in list(ulcer["Name"]))
#res_b[res_b['is_in_ulcer']== True]


# In[12]:


new_dataset = res_b[res_b['is_in_ulcer'] == True].copy()
# Assuming you have a DataFrame named df
worker_type_counts = new_dataset.groupby('Worker_type')['Name'].count().reset_index()
worker_type_counts.rename(columns={'Name': 'Total_Count_True'}, inplace=True)

worker_type_total = res_b.groupby('Worker_type').size().reset_index(name='Total_Count')

merged_df = worker_type_counts.merge(worker_type_total, on='Worker_type', suffixes=('_True', '_Total'))

merged_df['Percentage']=merged_df['Total_Count_True']/merged_df['Total_Count']


# In[13]:


# 假设你的 DataFrame 名称为 df
merged_df = merged_df.iloc[1:]

# 重置索引
merged_df.reset_index(drop=True, inplace=True)

# 输出结果
#print(merged_df)


# In[14]:


plt.figure(figsize=(10, 6))
plt.bar(merged_df['Worker_type'], merged_df['Percentage'])

for i, val in enumerate(merged_df['Percentage']):
    plt.text(i, val, f'{val:.2f}', ha='center', va='bottom')

# 添加标题和轴标签
plt.title('Percentage by Worker Type')
plt.xlabel('Worker Type')
plt.ylabel('Percentage')

# 显示柱状图
plt.show()


# In[15]:


# Function to determine severity
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


# In[16]:


birth_cp = birth.copy()
case_cp = case.copy()
ulcer_cp = ulcer.copy()
brad_cp = brad.copy()


# # Merge

# ## Merge ulcer + brad

# In[17]:


#Merge two tables
ulcer_b = ulcer.merge(brad, left_on='Name', right_on='Name')
ulcer_b = ulcer_b.dropna()

# Calculate the maximum allowed date (SOE Date + 60 days)
ulcer_b['MaxAllowedDate'] = ulcer_b['SOE'] + timedelta(days=60)

# Filter the rows where VisitDate is greater than SOEDate but not greater than 60 days
ulcer_b = ulcer_b[(ulcer_b['Visitdate'] >= ulcer_b['SOE']) & (ulcer_b['Visitdate'] <= ulcer_b['MaxAllowedDate'])]

# Drop the MaxAllowedDate column
ulcer_b = ulcer_b.drop(columns=['MaxAllowedDate'])
ulcer_b = ulcer_b.sort_values('Name', ascending = True)


# ## Merge ulcer + case

# In[18]:


ulcer_c = ulcer.merge(case, left_on='Name', right_on='Name')
ulcer_c = ulcer_c.dropna()


# Calculate the maximum allowed date (SOE Date + 60 days)
ulcer_c['MaxAllowedDate'] = ulcer_c['SOE'] + timedelta(days=60)

# Filter the rows where VisitDate is greater than SOEDate but not greater than 60 days
ulcer_c = ulcer_c[(ulcer_c['Start_of_Episode'] >= ulcer_c['SOE']) & (ulcer_c['Start_of_Episode'] <= ulcer_c['MaxAllowedDate'])]

# Drop the MaxAllowedDate column
ulcer_c = ulcer_c.drop(columns=['MaxAllowedDate'])
ulcer_c = ulcer_c.sort_values('Name', ascending = True)


# ## Merge brad + birthday

# In[19]:


#Merge two tables
brad_b = brad.merge(birth, left_on='Name', right_on='Name')
brad_b = brad_b.dropna()
brad_b['Age_as_of_visit'] = (brad_b['Visitdate'].dt.year - brad_b['DOB']).astype(int)


# ## Merge ulcer_b + birthday

# In[20]:


ulcer_bb = ulcer_b.merge(birth, left_on = 'Name', right_on = 'Name')
ulcer_bb['Age_as_of_ulcer'] = (ulcer_bb['SOE'].dt.year - ulcer_bb['DOB']).astype(int)


# # Export Processed Dataset

# In[21]:


# excel_bi = file_path + 'birth.xlsx'
# birth.to_excel(excel_bi, index=False)

# excel_ca = file_path + 'case.xlsx'
# case.to_excel(excel_ca, index=False)

# excel_br = file_path + 'brad.xlsx'
# brad.to_excel(excel_br, index=False)

# excel_ul = file_path + 'ulcer.xlsx'
# ulcer.to_excel(excel_ul, index=False)

# excel_uc = file_path + 'ulcer_c.xlsx'
# ulcer_c.to_excel(excel_uc, index=False)

# excel_bb = file_path + 'brad_b.xlsx'
# brad_b.to_excel(excel_bb, index=False)

# excel_ubb = file_path + 'ulcer_bb.xlsx'
# ulcer_bb.to_excel(excel_ubb, index=False)


# # Birthday EDA

# In[22]:


#brad_b
brad_b = brad_b.sort_values('Visitdate', ascending = True)
brad_b_sub = brad_b.drop_duplicates('Name',keep = 'first')

# Calculate the birth year and create 5-year bins
bins = np.arange(0, brad_b_sub['Age_as_of_visit'].max() + 5, 5)

# Create a histogram to count people in each 5-year bin
plt.figure(figsize=(12, 7))
plt.hist(brad_b_sub['Age_as_of_visit'], bins=bins, edgecolor='white', alpha=0.7)
plt.title('Count by 5-Year Birth Year Bins')
plt.xlabel('Birth Year Bins')
plt.ylabel('Count')

# Add labels on top of each bar
for i in range(len(bins) - 1):
    count = len(brad_b_sub[(brad_b_sub['Age_as_of_visit'] >= bins[i]) & (brad_b_sub['Age_as_of_visit'] < bins[i + 1])])
    plt.text((bins[i] + bins[i + 1]) / 2, count, str(count), ha='center', va='bottom')

# Set custom x-axis labels for the 5-year bins
plt.xticks(bins, rotation=45)

plt.tight_layout()
plt.show()


# In[23]:


#brad_b

ulcer_bb = ulcer_bb.sort_values('Visitdate', ascending = True)
ulcer_bb_sub = ulcer_bb.drop_duplicates('Name',keep = 'first')

# Calculate the birth year and create 5-year bins
bins = np.arange(10, ulcer_bb_sub['Age_as_of_ulcer'].max() + 5, 5)

# Create a histogram to count people in each 5-year bin
plt.figure(figsize=(12, 7))
plt.hist(ulcer_bb_sub['Age_as_of_ulcer'], bins=bins, edgecolor='white', alpha=0.7)
plt.title('Count by 5-Year Birth Year Bins')
plt.xlabel('Birth Year Bins')
plt.ylabel('Count')

# Add labels on top of each bar
for i in range(len(bins) - 1):
    count = len(ulcer_bb_sub[(ulcer_bb['Age_as_of_ulcer'] >= bins[i]) & (ulcer_bb_sub['Age_as_of_ulcer'] < bins[i + 1])])
    plt.text((bins[i] + bins[i + 1]) / 2, count, str(count), ha='center', va='bottom')

# Set custom x-axis labels for the 5-year bins
plt.xticks(bins, rotation=45)

plt.tight_layout()
plt.show()


# # Ulcer development  << Braden score

# In[24]:


#Wound development rate: For each braden score, how many of them is included in the ulcer dataset?

res_b = brad[["Name", "AssessmentAnswer"]]
res_b = res_b.drop_duplicates()
res_b['is_in_ulcer'] = res_b["Name"].apply(lambda x: x in list(ulcer_b["Name"]))
res_b[res_b['is_in_ulcer']== True].sort_values('Name',ascending = True)


# In[25]:


# brad['got_ulcer'] = brad["Name"].apply(lambda x: x in list(ulcer_b["Name"]))
# brad[brad['got_ulcer']== True].sort_values('Name',ascending = True)


# In[26]:


data1 = ulcer_b.drop(columns = "AssessmentQuestion")


# # Predictive Logistics model of Ulcer development from Brad Dataframe
# 

# In[27]:


# Drop any rows with missing values, assuming you've handled missing data appropriately
brad = brad.dropna()

# Selecting features (all columns except 'got_ulcer') and target variable

X = brad[['Severity', 'Worker_name', 'Worker_type', 'ServiceCode', 'AssessmentAnswer']]
y = brad['got_ulcer']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a transformer for categorical variables
categorical_features = ['Severity', 'Worker_name', 'Worker_type', 'ServiceCode', 'AssessmentAnswer']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor and logistic regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# Fit the model on the training data
model.fit(X_train, y_train)

categorical_features_names = model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(categorical_features)
feature_names = list(categorical_features_names) + list(X.select_dtypes(include=['float64', 'int64']).columns) + ['intercept']

# Create a dictionary to store coefficients
coef_dict = {}
for coef, feat in zip(model.named_steps['classifier'].coef_[0], feature_names):
    coef_dict[feat] = coef

# Convert the dictionary to a DataFrame
coefficients_df = pd.DataFrame(list(coef_dict.items()), columns=['Feature', 'Coefficient'])

# Sort the DataFrame by absolute values of coefficients (most significant to least significant)
coefficients_df['Abs_Coefficient'] = coefficients_df['Coefficient'].abs()
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False).reset_index(drop=True)

# Set option to display all rows
pd.set_option('display.max_rows', None)

# Print the sorted coefficients
print(coefficients_df[['Feature', 'Coefficient']])

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
# print(f"Accuracy: {accuracy:.2f}")
# print("\nConfusion Matrix:")
# print(conf_matrix)
# print("\nClassification Report:")
# print(classification_rep)


# # Predictive Linear Regression model of Braden score from ulcer_b Dataframe

# In[28]:


X = pd.get_dummies(ulcer_b[['Location', 'Type', 'Worker_type', 'ServiceCode', 'ServiceCode']])
y = ulcer_b['AssessmentAnswer']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

# print(f'R-squared: {r2}')
# print(f'Root Mean Squared Error: {rmse}')


# In[29]:


# brad.to_excel('brad.xlsx', index=False)
# ulcer_b.to_excel('ulcer_b.xlsx', index=False)


# In[30]:


unique_values_res_b = res_b['AssessmentAnswer'].unique()
perc_df = pd.DataFrame(columns=['AssessmentAnswer', 'Percentage'])
for value in unique_values_res_b:
  filtered_res_b = res_b[res_b['AssessmentAnswer'] == value]
  included_count = filtered_res_b['Name'].isin(ulcer_b['Name']).sum()
  total_count = len(filtered_res_b)
  percentage = round((included_count / total_count) * 100,2)
  perc_df = perc_df.append({'AssessmentAnswer': value, 'Percentage': percentage}, ignore_index=True)

perc_df = perc_df.sort_values(by='AssessmentAnswer')
perc_df


# In[31]:


perc_df['AssessmentAnswer'] = perc_df['AssessmentAnswer'].astype(int)

plt.figure(figsize=(12, 7))  # Adjust the figure size if needed
bars = plt.bar(perc_df['AssessmentAnswer'], perc_df['Percentage'], color='skyblue')
plt.xlabel('Braden Score')
plt.ylabel('Percentage')
plt.title('Likelihood of ulcer development of patients in different braden score')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.xticks(range(min(perc_df['AssessmentAnswer']), max(perc_df['AssessmentAnswer']) + 1, 1), rotation=0)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[32]:


# Get unique values from the 'AssessmentAnswer' column in ulcer_b
unique_values_ulcer_b = brad['AssessmentAnswer'].unique()

# An empty DataFrame to store the percentages
prev_df = pd.DataFrame(columns=['AssessmentAnswer', 'Prevalent_location', 'Percentage'])

# Iterate through each unique value in 'AssessmentAnswer'
for value in unique_values_ulcer_b:
    # Filter the DataFrame to include only rows with the current 'AssessmentAnswer' value
    filtered_df = ulcer_b[ulcer_b['AssessmentAnswer'] == value]

    # Calculate the count of occurrences for each location
    location_counts = filtered_df['Location'].value_counts()

    # Find the location with the highest count (most prevalent location)
    prevalent_location = location_counts.idxmax()

    # Calculate the percentage of the count of the most prevalent location
    # divided by the total count of rows for that assessment answer number
    total_count = len(filtered_df)
    percentage = (location_counts[prevalent_location] / total_count) * 100

    # Append the results to the DataFrame
    prev_df = prev_df.append({'AssessmentAnswer': value, 'Prevalent_location': prevalent_location, 'Percentage': percentage}, ignore_index=True)
    prev_df = prev_df.sort_values('AssessmentAnswer', ascending = True)
prev_df


# # Ulcer development << Case manager role

# In[33]:


# Create case_sub by dropping duplicates in 'case_manager_name' and keeping the first occurrence
case_sub = case.drop_duplicates(subset='case_manager_name', keep='first')
case_sub.reset_index(drop=True, inplace=True)
case_sub

# Filter the DataFrame to include only RN, PT, and OT case manager types
valid_types = ['RN', 'PT', 'OT']
filtered_case_sub = case_sub[case_sub['case_manager_type'].isin(valid_types)]

# Count the case manager types
type_counts = filtered_case_sub['case_manager_type'].value_counts().sort_values(ascending=False)

# An empty DataFrame to store the percentages
type_df = pd.DataFrame(columns=['case_manager_type', 'Percentage'])

# Calculate the percentage for each unique value in res_b
for value in valid_types:
    # Filter res_b for the current value and check if it's in the 'Name' column of ulcer
    filtered_res_c = filtered_case_sub[filtered_case_sub['case_manager_type'] == value]
    included_count = filtered_res_c['Name'].isin(ulcer['Name']).sum()
    total_count = len(filtered_res_c)
    percentage = round((included_count / total_count) * 100,2)
    type_df = type_df.append({'case_manager_type': value, 'Percentage': percentage}, ignore_index=True)

# Sort the DataFrame by 'AssessmentAnswer'
type_df = type_df.sort_values(by='Percentage', ascending = False)


# In[34]:


plt.figure(figsize=(7, 7))  # Adjust the figure size if needed
bars = plt.bar(type_df['case_manager_type'], type_df['Percentage'], color='skyblue')
plt.xlabel('case_manager_type')
plt.ylabel('Percentage')
plt.title('Likelihood of ulcer development of patients in under different case manager types')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}%', fontsize = 14, ha='center', va='bottom')

plt.tight_layout()
plt.show()


# # Visit Timespan << Braden score

# In[35]:


no_risk = list(brad[brad['Severity'] == 'No Risk']['Name'].unique())


# In[36]:


# Group by 'Name' and 'Visitdate' and aggregate into lists
brad_v = brad_b.groupby(['Name', 'Visitdate']).agg(list).reset_index()

# Sort the DataFrame by 'Visitdate' within each 'Name' group
brad_v.sort_values(by=['Name', 'Visitdate'], inplace=True)

# Group by 'Name' and select the first and last visit dates
brad_v['first_visit'] = brad_v.groupby('Name')['Visitdate'].transform('first')
brad_v['last_visit'] = brad_v.groupby('Name')['Visitdate'].transform('last')
brad_v['duration'] = (brad_v['last_visit'] - brad_v['first_visit']).dt.days

# Reset the index to have a clean DataFrame
brad_v.reset_index(inplace=True)


# In[37]:


# Merge 'brad_b' with 'brad_v' on the 'Name' column
brad_b = pd.merge(brad_b, brad_v[['Name', 'duration']], on='Name', how='left')


# In[47]:


brad_b2 = brad_b.drop_duplicates('Name', keep = 'first')

# Create a histogram with bins of 30 days each
plt.figure(figsize=(12, 7))  # Set the figure size
plt.hist(brad_b2['duration'], bins=range(0, brad_b2['duration'].max() + 30, 30), edgecolor='white', alpha=0.7)

# Add labels and title
plt.xlabel('Duration (days)')
plt.ylabel('Number of Patients')
plt.title('Distribution of Visit Durations')

# Set x-axis tick positions and labels to intervals of 365 days
tick_positions = range(0, brad_b2['duration'].max() + 365, 365)
tick_labels = [f'{year // 365} year' for year in tick_positions]

plt.xticks(tick_positions, tick_labels)

# Show the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# # Ulcer development << Ulcer Type

# In[39]:


# Group by 'Name', 'SOE', 'Location', and aggregate into lists
ulcer_t = ulcer_b.sort_values(by=['Name', 'SOE', 'Location'], ascending=[True, False, True])
ulcer_t = ulcer_t.groupby(['Name', 'SOE', 'Location']).agg(list).reset_index()

# Group by 'Name' and select the first and last ulcer type within the same location
ulcer_t['first_type'] = ulcer_t.groupby(['Name', 'Location'])['Type'].transform('first')
ulcer_t['last_type'] = ulcer_t.groupby(['Name', 'Location'])['Type'].transform('last')

# Calculate the 'diff' by iterating through rows and calculating the element-wise difference
ulcer_t['diff'] = ulcer_t.apply(lambda row: [last - first for first, last in zip(row['first_type'], row['last_type'])], axis=1)

# Reset the index to have a clean DataFrame
ulcer_t.reset_index(inplace=True)

# Filter rows where the 'diff' column contains non-zero values within the same location
ulcer_filtered = ulcer_t[ulcer_t['diff'].apply(lambda x: any(val != 0 for val in x))]


# In[40]:


# 根据 "Name" 和 "Location" 列进行分组，保留每个组的第一行
deduplicated_df = ulcer_t.drop_duplicates(subset=['Name', 'Location'])
deduplicated_df['diff_number'] = deduplicated_df['diff'].apply(lambda x: x[0] if x else None)


# In[41]:


# Calculate the counts for "Worse," "Better," and "Same"
worse_count = (deduplicated_df['diff_number'] > 0).sum()
better_count = (deduplicated_df['diff_number'] < 0).sum()
same_count = (deduplicated_df['diff_number'] == 0).sum()

# Create a bar graph
plt.figure(figsize=(8, 6))
plt.bar(['Same', 'Better', 'Worse'], [same_count, better_count, worse_count], color='skyblue')
plt.xlabel('Comparison',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.title('Patient getting better & worse',fontsize=16)
plt.text(-0.1, same_count + 10, str(same_count), fontsize=14, color='black')
plt.text(0.9, better_count + 10, str(better_count), fontsize=14, color='black')
plt.text(1.9, worse_count + 10, str(worse_count), fontsize=14, color='black')

# Set the font size for x-axis and y-axis labels
plt.xticks(fontsize=14)  # Adjust fontsize here for x-axis
plt.yticks(fontsize=14)  # Adjust fontsize here for y-axis

plt.show()


# In[42]:


# Calculate the counts for "Worse," "Better," and "Same"
worse_count = (deduplicated_df['diff_number'] > 0).sum()
better_count = (deduplicated_df['diff_number'] < 0).sum()
same_count = (deduplicated_df['diff_number'] == 0).sum()

# Create a pie chart
plt.figure(figsize=(6, 6))
labels = ['Worse', 'Better', 'Same']
sizes = [worse_count, better_count, same_count]
colors = ['skyblue', 'lightcoral', 'lightgreen']
explode = (0.1, 0.1, 0.1)  # Explode slices for emphasis, adjust as needed

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=explode,
        textprops={'fontsize': 14})

plt.title('Patient Getting Better & Worse & Same',fontsize=16)

plt.show()


# In[43]:


# Filter rows where diff_number is positive
positive_diff_df = deduplicated_df[deduplicated_df['diff_number'] > 0]

# Calculate the count of each positive diff_number value
positive_diff_counts = positive_diff_df['diff_number'].value_counts()

# Create a bar graph with annotations
plt.figure(figsize=(8, 6))
ax = positive_diff_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Type changed',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.title('Patients get worse',fontsize=16)

# Annotate each bar with its count
for i, count in enumerate(positive_diff_counts):
    ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=15)

plt.xticks(rotation=0)

# Set y-axis limits from 0 to 20
plt.yticks([0, 5, 10, 15, 20])

# Set the font size for x-axis and y-axis labels
plt.xticks(fontsize=14)  # Adjust fontsize here for x-axis
plt.yticks(fontsize=14)  # Adjust fontsize here for y-axis

plt.show()


# In[44]:


# Filter rows where diff_number is positive
positive_diff_df = deduplicated_df[deduplicated_df['diff_number'] < 0]

# Calculate the count of each positive diff_number value
positive_diff_counts = positive_diff_df['diff_number'].value_counts()

# Create a bar graph with annotations
plt.figure(figsize=(8, 8))
ax = positive_diff_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Type changed',fontsize=16)
plt.ylabel('Count',fontsize=16)
plt.title('Patients get better',fontsize=18)

# Annotate each bar with its count
for i, count in enumerate(positive_diff_counts):
    ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=16)

plt.xticks(rotation=0)

# Set y-axis limits from 0 to 20
plt.yticks([0, 5, 10, 15, 20,25])

# Set the font size for x-axis and y-axis labels
plt.xticks(fontsize=16)  # Adjust fontsize here for x-axis
plt.yticks(fontsize=16)  # Adjust fontsize here for y-axis

plt.show()

