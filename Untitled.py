#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib as pyplot


# In[4]:


import seaborn as sns


# In[5]:


df=pd.read_csv("retail_sales_dataset.csv")


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


null_values = df.isnull().sum()


# In[10]:


print(null_values)


# In[11]:


duplicate_rows = df[df.duplicated()]


# In[12]:


print(duplicate_rows)


# In[13]:


df.drop_duplicates(inplace=True)


# In[14]:


df.describe()


# In[15]:


df.shape


# In[16]:


df.head()


# In[17]:


mean_values_totalamount = df["Total Amount"].mean()


# In[18]:


print(mean_values_totalamount)


# In[19]:


median_values_totalamount=df["Total Amount"].median()


# In[20]:


mode_values_age=df["Age"].mode()[0]


# In[21]:


print(mode_values_age)


# In[22]:


mode_values_totalamount=df["Total Amount"].mode()[0]


# In[23]:


print(mode_values_totalamount)


# In[24]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[25]:


monthly_sales = df['Total Amount'].resample('M').sum()


# In[26]:


rolling_mean = monthly_sales.rolling(window=12).mean()


# In[27]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
monthly_sales.plot(label='Original Sales')
rolling_mean.plot(label='Trend (12-Month Rolling Mean)')
plt.title('Monthly Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[28]:


df.info()


# In[37]:


plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black') 
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Quantity')
plt.grid(True)  
plt.show()


# In[40]:


plt.figure(figsize=(10, 8))
correlation_matrix = df[['Total Amount', 'Quantity', 'Age']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[42]:


sales_by_category = df.groupby('Product Category')['Total Amount'].sum()

plt.figure(figsize=(8, 8))
plt.pie(sales_by_category, labels=sales_by_category.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Sales by Product Category')
plt.axis('equal')  
plt.show()


# In[43]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='Product Category', y='Total Amount', data=df)
plt.title('Distribution of Total Amount by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Amount')
plt.xticks(rotation=45) 
plt.tight_layout()  
plt.show()

plt.figure(figsize=(8, 6))
sns.violinplot(x='Gender', y='Quantity', data=df)
plt.title('Distribution of Quantity by Gender')
plt.xlabel('Gender')
plt.ylabel('Quantity')
plt.tight_layout()  
plt.show()


# In[44]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Product Category', y='Total Amount', data=df)
plt.title('Total Amount Sold by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Amount')
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x='Product Category', y='Quantity', data=df)
plt.title('Quantity Sold by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Quantity')
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()


# In[45]:


sales_by_category = df.groupby('Product Category')['Total Amount'].sum()


top_categories = sales_by_category.nlargest(3) 
print("Top-performing categories:", top_categories)


# In[46]:



sales_by_gender = df.groupby('Gender')['Total Amount'].sum()


plt.figure(figsize=(8, 6))
sns.barplot(x=sales_by_gender.index, y=sales_by_gender.values)
plt.title('Sales by Gender')
plt.xlabel('Gender')
plt.ylabel('Total Amount')
plt.show()


# In[ ]:




