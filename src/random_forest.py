# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:13:10 2019

@author: artur

Application of a random forest to the data set vectorized using a bag of words
model
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack
import numpy as np

file_path = '../data/samples/sample_train_unbalanced_red.csv'
f1 = open(file_path)
f2 = open(file_path)
f3 = open(file_path)
#Setting interactables for the labels and comments
labels = np.array([int(line.split('\t')[0]) for line in f1])
orig_comments = [line.split('\t')[1] for line in f2]
parent_comments = [line.split('\t')[9] for line in f3]
#setting vectorizer to remove stop words annd setting a maximal number of features
vectorizer = CountVectorizer(max_features = 500, stop_words = 'english')
#training and applaying vectorizer to original comments, then applaying it
#again to parent comments (so a position represents the same word on both vecore)
X_orig = vectorizer.fit_transform(orig_comments)
X_parent = vectorizer.transform(parent_comments)
#Concatenating the vectors so each row represents (original cooment, parent comment)
X = hstack((X_orig, X_parent))
#splitting
X_train, X_test, y_train, y_test = train_test_split(X, labels)
#setting up random forest
classifier = RandomForestClassifier()
#training classifier
classifier.fit(X_train, y_train)
#Getting metrics
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)
print("\n Train precision, recall, F1 and support:")
print(precision_recall_fscore_support(y_train, y_train_pred))
print("\n \n Test precision, recall, F1 and support:")
print(precision_recall_fscore_support(y_test, y_test_pred))
