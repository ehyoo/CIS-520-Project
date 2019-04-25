# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:44:59 2019

@author: artur
"""
from scipy.sparse import load_npz
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_validate
import json
import time
from datetime import timedelta
import pickle



def read_labels(path):
    Y = []
    with open(path) as f:
        l = f.readline()
        while l:
            Y.append(int(l[0]))
            l = f.readline()
    return Y


Train_labels_path = '../data/train-labels.csv'
Test_labels_path = '../data/test-labels.csv'
Train_features_path = '../data/features/Train_features.npz'
Test_features_path = '../data/features/Test_features.npz'


#Train_labels_path = '../data/train-labels_red.csv'
#Test_labels_path = '../data/test-labels_red.csv'
#Train_features_path = '../data/features/Train_features_red.npz'
#Test_features_path = '../data/features/Test_features_red.npz'


Y_train = read_labels(Train_labels_path)
X_train = load_npz(Train_features_path).tocsr()

deg = [1,2,3]
C = [1e-1, 1, 10, 100]
W = [75, 150, 225]

scores = {}
#f = open('../results/SVM_raw.json')
#scores = json.load(f)
#f.close()


metrics = ['precision', 'recall', 'f1']

t_0 = time.time()

for d in deg:
    for c in C:
        for w in W:
            if d ==1:
                model = LinearSVC(C = c, class_weight = {0 :1, 1: w})
            else:
                model = SVC(C = c, kernel = 'poly', degree = d, class_weight = {0 :1, 1: w})
            scores[(d, c, w)] = cross_validate(model, X_train, Y_train,scoring = metrics, n_jobs = -1, cv = 3 )
            time_str = str(timedelta(seconds = time.time() - t_0))
            print('CV scores for d = {}, c = {}, w = {}:\n {} \n Time since begining: {}\n'.format(d,c,w, scores[(d,c,w)], time_str))
            with open('../results/SVM_raw_d1.pickle', 'wb+') as f:            
                pickle.dump(scores,f)
            
    
    
