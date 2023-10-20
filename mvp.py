#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import sys
from datetime import datetime as dt, timedelta
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import streamlit as st


# In[ ]:


def upload_excel():
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file is not None:
        try:
            birth = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully. DataFrame 'birth' created.")
            return birth
        except Exception as e:
            st.error(f"Error: {e}")
    return None


# In[ ]:


birth['DOB'] = birth['DOB'].dt.year
birth['Name'] = birth['Name'].str.split('Last').str[1].str.split(', First').str.join('-')
birth = birth[birth['DOB'].notna()]
birth = birth.sort_values('DOB', ascending = True)

dup_birth = birth[birth['Name'].duplicated(keep=False)]
dup_birth = dup_birth.sort_values('Name', ascending = True)
print(len(birth))


# In[ ]:


def main():
    st.title("Streamlit App with Excel Upload")

    # Call the function to upload Excel file and get DataFrame
    birth = upload_excel()

    # Rest of your Streamlit app logic goes here
    if birth is not None:
        st.write("Preview of 'birth' DataFrame:")
        st.write(birth.head())

if __name__ == "__main__":
    main()

