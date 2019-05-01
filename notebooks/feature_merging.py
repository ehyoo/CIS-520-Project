#!/usr/bin/env python
# coding: utf-8

### Note: You need at least 16gb of ram to run this

# Merging the features.

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import scipy
import os


# Step 1: Loading in our data

# In[2]:


data_base_dir = '../data/features'


# In[3]:

print("Reading in our data...")
# First loading in our CSVs
pointedness_train_df = pd.read_csv(os.path.join(data_base_dir, "pointedness_train.csv"))
pointedness_test_df = pd.read_csv(os.path.join(data_base_dir, "pointedness_test.csv"))

synset_train_df = pd.read_csv(os.path.join(data_base_dir, "synset_train.csv"))
synset_test_df = pd.read_csv(os.path.join(data_base_dir, "synset_test.csv"))

frequency_train_df = pd.read_csv(os.path.join(data_base_dir, 'frequency_train.csv'))
frequency_test_df = pd.read_csv(os.path.join(data_base_dir, 'frequency_test.csv'))


# In[4]:


csv_feature_train_df = pointedness_train_df.join(synset_train_df).join(frequency_train_df)
csv_feature_test_df = pointedness_test_df.join(synset_test_df).join(frequency_test_df)


# In[5]:


csv_feature_train_matrix = csv_feature_train_df.values
csv_feature_test_matrix = csv_feature_test_df.values


# In[6]:


# Then loading in Artur's pattern work

pattern_train = scipy.sparse.load_npz(os.path.join(data_base_dir, "pattern_training.npz")).todense()
pattern_test = scipy.sparse.load_npz(os.path.join(data_base_dir, "pattern_test.npz")).todense()

print("Done.")
# In[7]:


# To help alleviate some memory pains
# Honestly if this 16gb monster can't handle it then idk lol
import gc

del pointedness_train_df
del pointedness_test_df

del synset_train_df
del synset_test_df

del frequency_train_df
del frequency_test_df

del csv_feature_train_df
del csv_feature_test_df

gc.collect()

print("Finished GC")
# In[ ]:

print("Making our training (please don't crash)")
X_train = np.concatenate((pattern_train, csv_feature_train_matrix), axis=1)
print("WOOHOO.")

# In[ ]:

print("Making X_train sparse...")
X_train = scipy.sparse.csc_matrix(X_train)
print("Done")

# In[19]:


del pattern_train
del csv_feature_train_matrix
gc.collect()


# In[12]:

print("Writing...")
scipy.sparse.save_npz(open("../data/features/X_train.npz", "wb+"), matrix=X_train)
print("Done")

# In[14]:


X_train.shape


# In[17]:


X_train = None
del X_train
gc.collect()


# In[20]:

print("Now doing testing...")
X_test = np.concatenate((pattern_test, csv_feature_test_matrix), axis=1)
X_test = scipy.sparse.csc_matrix(X_test)


# In[21]:


scipy.sparse.save_npz(open("X_test.npz", "wb+"), matrix=X_test)
print("Done")

# In[22]:


del pattern_test
del csv_feature_test_matrix
gc.collect()

