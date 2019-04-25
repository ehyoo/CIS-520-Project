# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:26:46 2019

@author: artur
"""

import pandas as pd
from scipy.sparse import load_npz, save_npz, hstack, csr_matrix, vstack
import random


def get_subset(X, Y, n):
    N = len(Y)
    rows = random.sample(range(0,N), int(n))
    X_red = []
    Y_red = []
    for r in rows:
        X_red.append(X.getrow(r))
        Y_red.append(Y[r])
    return vstack(X_red), Y_red

def write_Y(path,Y):
    f = open(path, 'w+')
    for y in Y:
        f.write('{}\n'.format(y))
    f.close()

def read_labels(path):
    Y = []
    with open(path) as f:
        l = f.readline()
        while l:
            Y.append(int(l[0]))
            l = f.readline()
    return Y

def get_red(n_test, n_train):
    direc = '../data/features/'
    X_test  = load_npz(direc + 'Test_features.npz').tocsr()
    X_train  = load_npz(direc + 'Train_features.npz').tocsr()
    Y_test = read_labels('../data/test-labels.csv')
    Y_train =read_labels('../data/train-labels.csv')
    X_train_red, Y_train_red = get_subset(X_train, Y_train, n_train)
    X_test_red, Y_test_red = get_subset(X_test, Y_test, n_test)
    save_npz(direc+'Test_features_red.npz', X_test_red)
    save_npz(direc+'Train_features_red.npz', X_train_red)
    write_Y('../data/test-labels_red.csv', Y_test_red )
    write_Y('../data/train-labels_red.csv', Y_train_red)


    
def join_features():
    direc = '../data/features/'
    Freq_test =  pd.read_csv(direc+'frequency_test.csv')
    Freq_train =  pd.read_csv(direc+'frequency_train.csv')
    Point_test =  pd.read_csv(direc+'pointedness_test.csv')
    Point_train =  pd.read_csv(direc+'pointedness_train.csv')
    Synset_test =  pd.read_csv(direc+'synset_test.csv')
    Synset_train =  pd.read_csv(direc+'synset_train.csv')
    
    Test_features = csr_matrix(pd.concat([Freq_test, Point_test, Synset_test], axis = 1).values)
    Train_features = csr_matrix(pd.concat([Freq_train, Point_train, Synset_train], axis = 1).values)
    
    Test_pattern = load_npz(direc + 'pattern_test.npz')
    Train_pattern = load_npz(direc + 'pattern_train.npz')
    
    X_test  = hstack([Test_features, Test_pattern])
    X_train = hstack([Train_features, Train_pattern])
    
    save_npz(direc+'Test_features.npz', X_test)
    save_npz(direc+'Train_features.npz', X_train)

get_red(1e4,1e6)






    
    


