#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import seaborn as sns


# In[4]:


import os


# In[5]:


print(os.listdir())


# In[6]:


df=pd.read_csv("AB_NYC_2019.csv")


# In[7]:


df


# In[8]:


df.isnull().sum()


# In[9]:


df.isnull().any()


# In[10]:


sns.heatmap(df.isnull(), yticklabels=False)


# In[11]:


null_values = df.isnull().sum()


# In[12]:


print(null_values)


# In[13]:


clean_df = df.dropna()


clean_df.reset_index(drop=True, inplace=True)


# In[14]:


df.isnull().sum()


# In[15]:


clean_df = df.drop_duplicates()

clean_df.reset_index(drop=True, inplace=True)


# In[19]:


from collections import Counter

def has_duplicates(data):
    counter = Counter(data)
    for count in counter.values():
        if count > 1:
            return True
    return False
if has_duplicates(df):
    print("Duplicate values exist in the dataset.")
else:
    print("No duplicate values found in the dataset.")


# In[ ]:




