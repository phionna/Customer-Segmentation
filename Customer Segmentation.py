#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Import and Reshape Data

# In[9]:


df = pd.read_csv('transactions_n100000.csv')


# In[10]:


df.head()


# In[11]:


df.shape


# In[12]:


#Separately get data grouped by ticket_id with their order timestamps and location data, and the other with item names and counts, and merge them together
grouped_df = df.groupby(by='ticket_id').first().drop(['item_name','item_count'],axis=1)
item_types = pd.pivot_table(df,index='ticket_id',columns='item_name',values='item_count').fillna(0)

new_df = grouped_df.merge(item_types,left_on='ticket_id',right_on='ticket_id')


# In[13]:


new_df.head()


# In[14]:


new_df.shape


# ## Feature Engineering

# In[15]:


new_df['order_timestamp'] = pd.to_datetime(new_df['order_timestamp'])
new_df['hour'] = new_df['order_timestamp'].dt.hour
new_df['day_of_week'] = new_df['order_timestamp'].dt.dayofweek


# In[16]:


new_df = new_df.drop(['lat','long','order_timestamp'],axis=1)


# In[17]:


#Based on Tableau Analytics:

def classify_location(loc):
    if loc in [1,3,5,8]:
        label = 'Chicago_City'
        #loc 8 is in Chicago prime location - Magnificent Mile
        #loc 5 is close to the central train station
    elif loc in [2,6]:
        label = 'University'
        #specifically, location 2 is near Northwestern, 6 is near UChic
    else:
        label = 'Suburban'
    return label

def classify_day_of_week(day):
    if day <= 4:
        label = 'weekday'
    else:
        label = 'weekend'
    return label

def classify_time_of_day(time):
    if (time >= 6) and (time <= 10):
        #6am-10am
        label = 'morning'
    elif (time >= 10) and (time <= 16):
        #10am- 4pm
        label = 'afternoon'
    elif (time >= 16) and (time <= 21):
        #4pm-9pm
        label = 'evening'
    else:
        #9pm-4am
        label = 'late_night'
    return label


# In[18]:


#new_df['location_type'] = new_df.location.apply(classify_location) decided not to use these labels as want to see if clustering algo picked it up for us
new_df['week_day'] = new_df.day_of_week.apply(classify_day_of_week)
new_df['time_of_day'] = new_df.hour.apply(classify_time_of_day)

new_df = new_df.drop(['day_of_week','hour'],axis=1)


# In[19]:


new_df.head()


# In[20]:


#One Hot Encoding using get_dummies

list_of_cols = ['location','week_day','time_of_day']

for var in list_of_cols:
    cat_list = pd.get_dummies(new_df[var])
    new_df = new_df.join(cat_list)

new_df = new_df.drop(list_of_cols,axis=1)


# In[21]:


new_df.head()


# In[22]:


#Ran the first time, found that weekday/weekend split not impt, so im removing it

new_df.drop(['weekday','weekend'],axis=1,inplace=True)


# ## Train-Test Split and Standardization

# In[23]:


from sklearn import model_selection

train,test = model_selection.train_test_split(new_df, test_size=0.3, random_state = 0)


# In[24]:


from sklearn import preprocessing


# In[25]:


#Use Min-max Scaler to scale data

Scaler = preprocessing.MinMaxScaler()
train_scaled = Scaler.fit_transform(train)
test_scaled = Scaler.transform(test)

train_scaled = pd.DataFrame(train_scaled, columns = train.columns,index= train.index)
test_scaled = pd.DataFrame(test_scaled,columns = test.columns,index= test.index)


# In[26]:


# Try with the standard scaler instead of Min-Max Scaler

Std_Scale = preprocessing.StandardScaler()
train_std_scaled = Std_Scale.fit_transform(train)
test_std_scaled = Std_Scale.transform(test)

train_std_scaled = pd.DataFrame(train_std_scaled, columns = train.columns,index= train.index)
test_std_scaled = pd.DataFrame(test_std_scaled,columns = test.columns,index= test.index)


# ## Clustering

# In[27]:


from sklearn.cluster import KMeans


# In[28]:


#Cluster with min-max scaled data
kmeans = KMeans(n_clusters=3, random_state=0).fit(train_scaled)


# In[29]:


columns = train_scaled.columns
centers = kmeans.cluster_centers_

pd.DataFrame(data=centers,columns=columns)


# In[30]:


#Proportion of clusters on train data
labels = pd.DataFrame(data= kmeans.labels_)
print(labels[0].value_counts(normalize=True))


# In[31]:


#Cluster with normalized scaled data

kmeans_std = KMeans(n_clusters=3, random_state=0).fit(train_std_scaled)

labels = pd.DataFrame(data= kmeans_std.labels_)
print(labels[0].value_counts(normalize=True))


# ## Validation

# In[32]:


#plotting elbow graph, using normalized scaled data
Sum_of_squared_distances = []
K = range(1,10)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(train_std_scaled)
    Sum_of_squared_distances.append(km.inertia_)


# In[33]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[34]:


#Look at proportion of clusters on test data
labels = pd.DataFrame(data= kmeans_std.predict(test_std_scaled))
labels[0].value_counts(normalize=True)


# ## Looking at the Mean Characteristics

# In[35]:


#Transform min-max scaled data back
transformed = Scaler.inverse_transform(train_scaled)

train_scaled_transformed = pd.DataFrame(transformed, columns = train.columns,index= train.index)

#Putting Labels back into DF
train_scaled_transformed['cluster_labels'] = kmeans.labels_


# In[36]:


train_scaled_transformed.groupby(by='cluster_labels').mean()


# In[39]:


#Transform normalized scaled data back
transformed = Std_Scale.inverse_transform(train_std_scaled)

train_scaled_transformed = pd.DataFrame(transformed, columns = train.columns,index= train.index)

#Putting Labels back into DF
train_scaled_transformed['cluster_labels'] = kmeans_std.labels_


# In[40]:


train_scaled_transformed.groupby(by='cluster_labels').mean()


# In[41]:


#Both ways of standardizing yield quite similar results, maybe min-max is better in splitting the times of the day

