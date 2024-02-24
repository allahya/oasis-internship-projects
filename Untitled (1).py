#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


df=pd.read_csv("creditcard.csv")


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


from sklearn.linear_model import LogisticRegression


# In[8]:


from sklearn.metrics import accuracy_score, classification_report


# In[9]:


df.tail()


# In[10]:


df.isnull()


# In[11]:


df.info()


# In[12]:


df['Class'].value_counts()


# In[13]:


legit=df[df.Class==0]
fraud=df[df.Class==1]


# In[14]:


print(legit.shape)
print(fraud.shape)


# In[15]:


legit.Amount.describe()


# In[16]:


df.groupby('Class').mean()


# In[17]:


legit_sample=legit.sample(n=492)


# In[18]:


newd=pd.concat([legit_sample,fraud],axis=0)


# In[19]:


newd.head()


# In[20]:


newd['Class'].value_counts()


# In[21]:


newd.groupby('Class').mean()


# In[22]:


X = newd.drop(columns='Class', axis=1)
Y = newd['Class']


# In[23]:


print(X)
print(Y)


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[25]:


model = LogisticRegression()


# In[26]:


model.fit(X_train, Y_train)


# In[27]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[28]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[29]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[30]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# In[ ]:




