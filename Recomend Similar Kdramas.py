#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Import package

# In[3]:


#Modules for EDA
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

#Modules for ML(Recommendation)
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


df = pd.read_csv('top100_kdrama.csv')
df.head()


# In[40]:


synopsis = df['Synopsis']
synopsis.head()


# ## Buat variabel terpisah khusus untuk data judul film

# In[41]:


kdrama_names = df[['Name']]
kdrama_names.head()


# ## seleksi fitur yang digunakan untuk Rekomendasi

# In[42]:


cols_for_recommend = ['Year of release', 'Number of Episode', 'Network', 'Duration', 'Content Rating', 'Rating']
df = df[cols_for_recommend]
df.head(10)


# ## Feature Engineering

# ## Menghapus nilai Duplikat di kolom Network

# In[45]:


networks = []
[networks.append(list(set(network.replace(' ','').split(',')))[0]) for network in df['Network']]


# In[46]:


df['Network'] = networks
df['Network'].unique()


# ## Network dan Total KDramas

# In[47]:


plt.figure(figsize=(7,7))

df['Network'].value_counts().plot(kind='barh')

plt.gca().invert_yaxis()
plt.title("Networks of Kdramas.")
plt.xlabel('Frequency')
plt.show()

df['Network'].value_counts()


# ## Mengelompokkan  OCN dan ViKi ke kategori 'others' karena jumlah yang  sedikit

# In[48]:


df['Network'].replace(['OCN','Viki'],['Others','Others'],inplace=True)


# In[49]:


df['Duration'] = df['Duration'].str.replace('[A-Za-z]\D+','',regex=True)
df['Duration'].head()


# In[50]:


df['Duration'] = df['Duration'].str.replace(' ','',regex=True)
df['Duration'] = pd.to_numeric(df['Duration'])
df['Duration'].head()


# In[51]:


plt.figure(figsize=(7,7))
sns.histplot(data=df['Duration'])
plt.title('Duration in minutes.')
plt.show()


# ## Content Rating

# In[52]:


plt.figure(figsize=(7,7))
df['Content Rating'].value_counts().plot(kind='pie',autopct='%.2f%%')
plt.title("Content Rating")
plt.show()


# In[53]:


df['Content Rating'].value_counts()


# # Rating dan Content Rating

# In[54]:


sns.histplot(data=df[['Rating','Content Rating']],x='Rating',hue='Content Rating')
plt.show()


# In[55]:


df[['Rating']].describe()


# ## One Hot Encoding

# In[56]:


cols_to_encode = ['Network','Content Rating']
dummies = pd.get_dummies(df[cols_to_encode],drop_first=True)
dummies.head()


# In[57]:


df


# In[58]:


dummies.columns


# In[59]:


df.drop(cols_to_encode, axis=1,inplace=True)
df.head()


# ## Feature Scaling

# In[60]:


scale = MinMaxScaler()
scalled = scale.fit_transform(df)


# In[61]:


i=0
for col in df.columns:
    df[col] = scalled[:,i]
    i += 1


# In[28]:


df.head()


# In[62]:


new_df = pd.concat([df, dummies],axis=1)
new_df


# In[30]:


new_df.head()


# In[31]:


kdrama_names['Name'].loc[23]='kingdom'


# In[63]:


new_df.index = [drama for drama in kdrama_names['Name']]
synopsis.index = [drama for drama in kdrama_names['Name']]


# In[117]:


synopsis


# In[65]:


new_df.head()


# In[66]:


def getRecommendation_dramas_for(drama_name,no_of_recommend=5,get_similarity_rate=False):
    
    kn = NearestNeighbors(n_neighbors=no_of_recommend+1,metric='manhattan')
    kn.fit(new_df)
    
    distances, indices = kn.kneighbors(new_df.loc[drama_name])
    
    print(f'Similar K-Dramas for "{drama_name[0]}":')
    nearest_dramas = [kdrama_names.loc[i][0] for i in indices.flatten()][1:]
    if not get_similarity_rate:
        return nearest_dramas
    sim_rates = []
    synopsis_ = []
    for drama in nearest_dramas:
        synopsis_.append(synopsis.loc[drama][0])
        sim = cosine_similarity(new_df.loc[drama_name],[new_df.loc[drama]]).flatten()
        sim_rates.append(sim[0])
    recommended_dramas = pd.DataFrame({'Recommended Drama':nearest_dramas,'Similarity':sim_rates,'Synopsis':synopsis_})
    recommended_dramas.sort_values(by='Similarity',ascending=True)
    return recommended_dramas


# ## Prediksi Rekomendasi Drama 

# In[67]:


rd1 = kdrama_names.loc[14]
rd1


# In[68]:


getRecommendation_dramas_for(rd1,no_of_recommend=5)


# In[ ]:




