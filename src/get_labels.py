# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:13:44 2019

@author: artur
"""

def get_labels(sample_path, write_path):
    ft = open(write_path, 'w+') 
    with open(sample_path) as f:
        line = f.readline()
        while line:
            lbl = line[0]
            ft.write(lbl+'\n')
            line = f.readline()
    ft.close()


        
sample_path = '../data/samples/test-unbalanced.csv'
write_path = '../data/test-labels.csv'
get_labels(sample_path, write_path)
sample_path = '../data/samples/train-unbalanced.csv'
write_path = '../data/train-labels.csv'
get_labels(sample_path, write_path)
