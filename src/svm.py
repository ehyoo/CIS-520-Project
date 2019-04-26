# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:44:59 2019

@author: artur
"""
from scipy.sparse import load_npz
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, f1_score, classification_report, roc_auc_score
import time
from datetime import timedelta
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MaxAbsScaler

tol = 1e-3
loss = 'log'
penalty = 'l1' 
 
def read_labels(path):
    Y = []
    with open(path) as f:
        l = f.readline()
        while l:
            Y.append(int(l[0]))
            l = f.readline()
    return np.array(Y)


def get_CV_data(X, Y, Alpha, verbose = False):
    global tol
    global loss
    global penalty
    t_0 = time.time()
    skf = StratifiedKFold(n_splits = 3)
    auc_scores = []
    i = 0
    for a in Alpha:
        model = SGDClassifier(loss = loss, alpha = a, class_weight = 'balanced', penalty = penalty, n_jobs = -1, tol =  tol)
        auc_scores.append(0)
        for train_index, test_index in skf.split(X,Y):
            model.fit(X[train_index], Y[train_index])
            scores = model.decision_function(X[test_index])
            roc_auc = roc_auc_score(Y[test_index],scores)
            auc_scores[i] = auc_scores[i]  + roc_auc/3
        if verbose:
            time_str = str(timedelta(seconds = time.time() - t_0))
            print('CV for alpha = {}\n AUC score = {} Time since begining: {}\n'.format(a,auc_scores[i], time_str))
        i += 1
    return auc_scores
    
def select_threshold(X, Y, a):
    global tol
    global loss
    global penalty
    skf = StratifiedKFold(n_splits = 3)
    model = SGDClassifier(loss = loss, alpha = a, class_weight = 'balanced', penalty = penalty, n_jobs = -1,tol = tol)
    thld = 0
    mean_f1 = 0
    for train_index, test_index in skf.split(X,Y):
        model.fit(X[train_index], Y[train_index])
        scores = model.decision_function(X[test_index])
        fpr, tpr, thresholds = roc_curve(Y[test_index], scores, pos_label = 1)
        f1 = []
        #thld_range = thresholds
        thld_range = np.linspace(thresholds[0], thresholds[-1], 50)
        for t in thld_range:
            f1.append(f1_score(Y[test_index],(scores > t).astype(int)))
        best_f1 = max(f1)
        best_t = thld_range[f1.index(best_f1)]
        thld = thld + best_t/3
        mean_f1 = mean_f1 + best_f1/3
    return thld, mean_f1

def evaluate(X_train, Y_train, X_test, Y_test, a, thld, plot = False, plot_path = '../results/Test_ROC.png'):
    global tol
    global loss
    global penalty
    model = SGDClassifier(loss = loss, alpha = a, class_weight = 'balanced', penalty = penalty, n_jobs = -1, tol = tol)
    model.fit(X_train, Y_train)
    
    train_scores = model.decision_function(X_train)
    Y_train_pred = (train_scores > thld).astype(int)
    train_report = classification_report(Y_train, Y_train_pred, digits = 3)
    print('Train report:\n{}'.format(train_report))

    test_scores = model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, test_scores, pos_label = 1)
    Y_test_pred = (test_scores > thld).astype(int)
    test_report = classification_report(Y_test, Y_test_pred, digits = 3)
    print('test report:\n{}'.format(test_report))
    if(plot):
        plt.plot(fpr, tpr)
        plt.plot(fpr,fpr, linestyle = ':', color = 'k' )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(plot_path)
        plt.show()
    ROC_data =(fpr, tpr, thld)    
    return train_report, test_report, ROC_data, model
    

def interpret_coef(coef, k):
    variable_names = json.load('../data/features/features_names.json')
    arg_sorted = np.argsort(-np.abs(coef))
    return [variable_names[arg_sorted[i]] for i in range(0,k)]
    
    


def experiment(Alpha, X_train, Y_train, X_test, Y_test, data_path, param_plot_path, roc_plot_path, model_path):
    data = {'Alplha': Alpha}
    print('Geting AUC scores')
    auc_scores = get_CV_data(X_train, Y_train, Alpha )
    best_auc = max(auc_scores)
    best_alpha = Alpha[auc_scores.index(best_auc)]
    data['auc_scores'] = auc_scores
    data['best_auc'] = best_auc
    data['best_alpha'] = best_alpha
    plt.plot(Alpha,auc_scores)
    plt.xlabel('Regulatization')
    plt.ylabel('Mean cross-validated AUC')
    plt.savefig(param_plot_path)
    plt.show()
    
    print('Selecting Threshold')    
    thld, CV_f1 = select_threshold(X_train, Y_train, best_alpha)
    data['thld'] = thld
    data['CV_f1'] = CV_f1
    
    print('Evaluating model')
    train_report, test_report, ROC_data, model = evaluate(X_train, Y_train, X_test, Y_test, best_alpha, thld, plot = True, plot_path = roc_plot_path)
    data['train_report'] = train_report
    data['test_report'] = test_report
    data['ROC_data'] = ROC_data
    data['model_coef'] = model.coef_
    
    print('Geting best features')
    arg_sorted = np.argsort(-np.abs(model.coef_))
    data['best_features'] = arg_sorted[0:10]
    #with open(data_path, 'wb+' ) as f:
    #    pickle.dump(data, f)
     
    with open(model_path, 'wb+' ) as f:
        pickle.dump(model, f)
        
    with open(data_path, 'wb+' ) as f:
        pickle.dump(data, f)
        
    return data
    
    
def get_data(balanced = False):
    if balanced:        
        Train_labels_path = '../data/features/y_train_balanced.npy'
        Train_features_path = '../data/features/X_train_balanced.npz'
        
        Test_labels_path = '../data/features/y_test_balanced.npy'
        Test_features_path = '../data/features/X_test_balanced.npz'
        
    else:
        Train_labels_path = '../data/features/y_train_full.npy'
        Train_features_path = '../data/features/X_train_full.npz'
        
        Test_labels_path = '../data/features/y_test_full.npy'
        Test_features_path = '../data/features/X_test_full.npz'

    Scaler = MaxAbsScaler()
    X_train = Scaler.fit_transform(load_npz(Train_features_path).tocsr())
    Y_train = np.load(Train_labels_path).ravel()
    
    X_test = Scaler.transform(load_npz(Test_features_path).tocsr())
    Y_test = np.load(Test_labels_path).ravel()
    
    return X_train, Y_train, X_test, Y_test

Alpha = np.linspace(1e-9,.5,10)

data_path = '../results/SGD_data_b.pickle'
param_plot_path = '../results/SGD_AUC_param_b.png' 
roc_plot_path = '../results/SGD_Test_ROC_b.png'
model_path = '../results/SGD_model_b.png' 

X_train, Y_train, X_test, Y_test  = get_data(balanced = True)
data_balanced = experiment(Alpha, X_train, Y_train, X_test, Y_test, data_path, param_plot_path, roc_plot_path, model_path)

data_path = '../results/SGD_data.pickle'
param_plot_path = '../results/SGD_AUC_param.png' 
roc_plot_path = '../results/SGD_Test_ROC.png'
model_path = '../results/SGD_model.png' 

X_train, Y_train, X_test, Y_test  = get_data(balanced = False)
data_unbalanced = experiment(Alpha, X_train, Y_train, X_test, Y_test, data_path, param_plot_path, roc_plot_path, model_path)





            
    
    
