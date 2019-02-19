
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os
import ast


# In[2]:


#Initial loading of data
os.chdir('C:\\Users\\mark.sinclair\\Documents\\kaggle\\movies')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_id = train['id']
test_id = test['id']

all_data = train.append(test, sort=True)


# In[3]:


# Starting some explatory analysis
print("Length of all_data")
print(len(all_data))


# In[4]:


# Checking prevalence of nulls
print(all_data.isna().sum())
# Collection, homepage, tagline have the three largest prevalence of nulls. Interestingly 3 have null titles.
# There are 0 null budget but need to check the 0 value.


# In[5]:


print('number of 0 for budgets')
print(len(all_data[all_data.budget == 0]))
# 2023 records have a value of 0 for the budget

print('number of 0 in budget for each train and test')
print(len(train[train.budget == 0]))
print(len(test[test.budget == 0]))
# 812 in the training dataset
# 1211 in the test dataset


# In[6]:


# Histogram of the budgets
plt.hist([all_data.budget], bins='auto')


# In[7]:


# Looking at the histogram of the log of budgets

# creating the log budget variable
log_budget=[]
for i in range(0,len(all_data["budget"])):
    if all_data.iloc[i,2] == 0:
        log_budget.append(0)
    else:
        log_budget.append(math.log10(all_data.iloc[i,2]))

all_data['log_budget'] = log_budget        
        
# plotting the histogram
plt.hist(all_data['log_budget'], bins='auto')


# In[8]:


# Plotting of the log budget against the log revenue (where known)

train_log_budget=[]
for i in range(0,len(train["budget"])):
    if train.iloc[i,2] == 0:
        train_log_budget.append(0)
    else:
        train_log_budget.append(math.log10(train.iloc[i,2]))
        
train_log_revenue=[]
for i in range(0,len(train["revenue"])):
    if train.iloc[i,22] == 0:
        train_log_revenue.append(0)
    else:
        train_log_revenue.append(math.log10(train.iloc[i,22]))
        
train['log_budget'] = train_log_budget
train['log_revenue'] = train_log_revenue

plt.scatter(train['log_budget'], train ['log_revenue'])
plt.xlabel('Log_budget')
plt.ylabel('log_revenue')


# In[9]:


# Only graphing where the budget is > 0
train_graph = train[train['log_budget'] > 0]

plt.scatter(train_graph['log_budget'], train_graph['log_revenue'])
plt.xlabel('Log_budget')
plt.ylabel('log_revenue')


# In[10]:


# Number of records have null for cast and crew
print(all_data[all_data.cast.isna()].head())


# In[11]:


print(all_data[all_data.cast.isna()].head())


# In[12]:


# records where cast is null but crew is not null
print(all_data[(all_data.cast.isna()) & (all_data.crew.notnull())])


# In[13]:


# records where crew is null but cast is not null
print(len(all_data[(all_data.crew.isna()) & (all_data.cast.notnull())]))
print('examples')
print(all_data[(all_data.crew.isna()) & (all_data.cast.notnull())].head())


# In[14]:


# Getting the number of cast in each movie
# Setting the dictionary for the str columns
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df
        
train = text_to_dict(train)
test = text_to_dict(test)


# In[15]:


# Number of cast in each film
train['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts()

train['num_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)
train['num_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)

test['num_cast'] = test['cast'].apply(lambda x: len(x) if x != {} else 0)
test['num_crew'] = test['crew'].apply(lambda x: len(x) if x != {} else 0)


# In[16]:


#Scatter plot of the number of cast against crew

plt.scatter(train['num_cast'], train['num_crew'])
plt.xlabel('num_cast')
plt.ylabel('num_crew')


# In[17]:


# Scatterplot of cast and crew against log budget and revenue

plt.figure(figsize=(8,8))

plt.subplot(2,2,1)
plt.scatter(train['num_cast'], train['log_budget'])
plt.xlabel('num_cast')
plt.ylabel('log_budget')

plt.subplot(2,2,2)
plt.scatter(train['num_cast'], train['log_revenue'])
plt.xlabel('num_cast')
plt.ylabel('log_revenue')

plt.subplot(2,2,3)
plt.scatter(train['num_crew'], train['log_budget'])
plt.xlabel('num_crew')
plt.ylabel('log_budget')

plt.subplot(2,2,4)
plt.scatter(train['num_crew'], train['log_revenue'])
plt.xlabel('num_crew')
plt.ylabel('log_revenue')


# In[18]:


# Is there a relationship between the number of cast and crew and the budget of the movie? 
# If successful this will allow budgets = 0 to get a modelled budget


