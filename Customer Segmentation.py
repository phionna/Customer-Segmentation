#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Import and Reshape Data

# In[3]:


df = pd.read_csv('transactions_n100000.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


grouped_df = df.groupby(by='ticket_id').first().drop(['item_name','item_count'],axis=1)


# In[8]:


grouped_df


# In[9]:


item_types = pd.pivot_table(df,index='ticket_id',columns='item_name',values='item_count').fillna(0)


# In[140]:


new_df = grouped_df.merge(item_types,left_on='ticket_id',right_on='ticket_id')


# In[141]:


new_df.head()


# In[142]:


new_df.shape


# ## Feature Engineering

# In[143]:


new_df['order_timestamp'] = pd.to_datetime(new_df['order_timestamp'])
new_df['hour'] = new_df['order_timestamp'].dt.hour
new_df['day_of_week'] = new_df['order_timestamp'].dt.dayofweek


# In[144]:


new_df = new_df.drop(['lat','long','order_timestamp'],axis=1)


# In[145]:


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


# In[146]:


#new_df['location_type'] = new_df.location.apply(classify_location)
new_df['week_day'] = new_df.day_of_week.apply(classify_day_of_week)
new_df['time_of_day'] = new_df.hour.apply(classify_time_of_day)

new_df = new_df.drop(['day_of_week','hour'],axis=1)


# In[147]:


new_df.head()


# In[148]:


#One Hot Encoding

list_of_cols = ['location','week_day','time_of_day']

for var in list_of_cols:
    cat_list = pd.get_dummies(new_df[var])
    new_df = new_df.join(cat_list)

new_df = new_df.drop(list_of_cols,axis=1)


# In[149]:


new_df.head()


# In[150]:


new_df.describe()


# In[151]:


#Ran the first time, found that weekday/weekend split not impt, so im removing it

new_df.drop(['weekday','weekend'],axis=1,inplace=True)


# ## Train-Test Split and Standardization

# In[152]:


from sklearn import model_selection

train,test = model_selection.train_test_split(new_df, test_size=0.3, random_state = 0)


# In[153]:


from sklearn import preprocessing


# In[154]:


Scaler = preprocessing.MinMaxScaler()
train_scaled = Scaler.fit_transform(train)
test_scaled = Scaler.transform(test)


# In[155]:


train_scaled = pd.DataFrame(train_scaled, columns = train.columns,index= train.index)
test_scaled = pd.DataFrame(test_scaled,columns = test.columns,index= test.index)


# In[180]:


# Try with the standard scaler instead of Min-Max Scaler

Std_Scale = preprocessing.StandardScaler()
train_std_scaled = Std_Scale.fit_transform(train)

train_std_scaled = pd.DataFrame(train_std_scaled, columns = train.columns,index= train.index)


# ## Clustering

# In[157]:


from sklearn.cluster import KMeans


# In[181]:


kmeans = KMeans(n_clusters=3, random_state=0).fit(train_std_scaled)


# In[182]:


columns = train_scaled.columns
centers = kmeans.cluster_centers_

pd.DataFrame(data=centers,columns=columns)


# In[183]:


labels = pd.DataFrame(data= kmeans.labels_)
print(labels[0].value_counts(normalize=True))


# ## Validation

# In[176]:


Sum_of_squared_distances = []
K = range(1,10)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(train)
    Sum_of_squared_distances.append(km.inertia_)


# In[177]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[178]:


print(Sum_of_squared_distances)


# In[184]:


labels = pd.DataFrame(data= kmeans.predict(test_scaled))
labels[0].value_counts(normalize=True)


# ## Looking at the Mean Characteristics

# In[166]:


transformed = Scaler.inverse_transform(train_scaled)

train_scaled_transformed = pd.DataFrame(transformed, columns = train.columns,index= train.index)


# In[167]:


#Putting Labels back into DF
train_scaled_transformed['cluster_labels'] = kmeans.labels_


# In[168]:


train_scaled_transformed.head()


# In[169]:


train_scaled_transformed.groupby(by='cluster_labels').mean()


# In[186]:


transformed = Std_Scale.inverse_transform(train_std_scaled)

train_scaled_transformed = pd.DataFrame(transformed, columns = train.columns,index= train.index)


# In[187]:


#Putting Labels back into DF
train_scaled_transformed['cluster_labels'] = kmeans.labels_


# In[188]:


train_scaled_transformed.groupby(by='cluster_labels').mean()


# In[189]:


#Both ways of standardizing yield quite similar results, maybe min-max is better in splitting the times of the day

