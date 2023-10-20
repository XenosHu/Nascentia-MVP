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

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from scipy.stats import ttest_ind
# from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


birth['DOB'] = birth['DOB'].dt.year
birth['Name'] = birth['Name'].str.split('Last').str[1].str.split(', First').str.join('-')
birth = birth[birth['DOB'].notna()]
birth = birth.sort_values('DOB', ascending = True)
birth


# In[ ]:


dup_birth = birth[birth['Name'].duplicated(keep=False)]
dup_birth = dup_birth.sort_values('Name', ascending = True)
print(len(birth))

