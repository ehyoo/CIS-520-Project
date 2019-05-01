# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:51:22 2019

@author: artur
"""

import numpy as np
import regex
from scipy.sparse import csr_matrix, save_npz, hstack
import time
from datetime import timedelta
import json
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd

class PatternFeatures:
    def __init__(self, HFW, min_ns_freq = 1e-3, min_s_freq = 1e-3, min_rel_freq = 1.5 ):
        self.min_ns_freq = min_ns_freq
        self.min_s_freq = min_s_freq
        self.log_min_rel_freq = abs(np.log(min_rel_freq))
        self.HFW = HFW
        self.num_patterns = 0
        self.patterns, self.reg = get_masks()#dictionary of dictionaries
    
    def Fit_transform(self, lines, verbose = False): 
        data = []
        t_0 = time.time()
        i = 0
        total_pos = 0
        #obtaining patterns
        for l in lines:
            total_pos += l[0]
            for k in (1,2):               
                d = self.get_patterns(l[k])
                for m in d:
                    for words in d[m]:
                        if not words in self.patterns[m]:
                            self.patterns[m][words] = self.num_patterns
                            data.append(Pattern_data(m, words, l[0], d[m][words] ,i, k-1))
                            self.num_patterns += 1
                        else:
                            n = self.patterns[m][words]
                            data[n].update(l[0],d[m][words], i, k-1)
            i += 1
            if(verbose & (i%1e6 == 0)):
                time_str = str(timedelta(seconds = time.time() - t_0))
                print('Line numer {}(10^6) fitted.\n OC:{} PC:{} \n Running time: {}\n'.format(i/1e6,l[1], l[2], time_str))
        total_comments = i
        
        #cutting patterns and reassigning ids
        j = 0
        while j < len(data):
            if self.cut(data[j], total_comments, total_pos):
                del self.patterns[data[j].mask][data[j].words]
                del data[j]
            else:
                self.patterns[data[j].mask][data[j].words] = j
                j += 1
        self.num_patterns = j
        #Creating matrices
        comm_row_ind = []
        comm_col_ind = []
        comm_occ=[]
        parent_row_ind = []
        parent_col_ind = []
        parent_occ=[]
        for i in range(0, self.num_patterns):
            comm_occ = comm_occ + data[i].original_occ
            comm_row_ind = comm_row_ind + data[i].original_instances
            comm_col_ind = comm_col_ind + [i]*len(data[i].original_instances)
            
            parent_occ = parent_occ + data[i].parent_occ
            parent_row_ind = parent_row_ind + data[i].parent_instances
            parent_col_ind = parent_col_ind + [i]*len(data[i].parent_instances)
        
        M_comm = csr_matrix((comm_occ,(comm_row_ind, comm_col_ind)), shape = (total_comments,self.num_patterns))
        M_parent = csr_matrix((parent_occ,(parent_row_ind, parent_col_ind)), shape = (total_comments,self.num_patterns))
        return M_comm, M_parent
        
    def get_patterns(self, comment):
        tok, is_word = tokenize(comment)
        rep = ''
        for i in range(0, len(tok)):
            if not is_word[i]:
                rep = rep + '+'
            else:
                if (tok[i] in self.HFW):
                    rep = rep + '1'
                else:
                    rep = rep + '0'
        d = {}
        matches = self.reg.finditer(rep, overlapped = True)
        for m in matches:
            mask = m.group(0)
            ind = [i for i in range(m.span()[0],m.span()[1]) if m.group(0)[i - m.span()[0]] == '1']
            words = tuple([tok[i] for i in ind])
            if mask in d:
                if words in d[mask]:                  
                    d[mask][words] = d[mask][words]+1
                else:
                    d[mask][words] = 1
            else:
                d[mask] = {words: 1}           
        return d
                              
    def cut(self, pattern, T, T_pos):
        ns_freq = pattern.count[0]/(T-T_pos)
        s_freq =  pattern.count[1]/T_pos
        freq_crit = (ns_freq > self.min_ns_freq)|(s_freq > self.min_s_freq)
        log_rel_freq = np.log(ns_freq) - np.log(s_freq)
        return (not freq_crit)|(abs(log_rel_freq) < self.log_min_rel_freq)
        
    def Transform(self, line):
        row_ind = [[],[]]
        col_ind = [[],[]]
        occ = [[],[]]
        i = 0
        for l in line:
            for k in (0,1):
                d = self.get_patterns(l[k])
                for m in d:
                    for words in d[m]:
                        if words in self.patterns[m]:
                            row_ind[k].append(i)
                            col_ind[k].append(self.patterns[m][words])
                            occ[k].append(d[m][words])
            i +=1
        M_comm = csr_matrix((occ[0],(row_ind[0], col_ind[0])), shape = (i,self.num_patterns))
        M_parent = csr_matrix((occ[1],(row_ind[1], col_ind[1])), shape = (i,self.num_patterns))
        return M_comm, M_parent
            
    def dump(self,file):
        D = {}
        for m in self.patterns:
            words  = {}
            for t in self.patterns[m]:
                words[' '.join(t)] = self.patterns[m][t]
            D[m] = words
        HFW = []
        for w in self.HFW:
            HFW.append(w)    
        data = { 'min_ns_freq' : self.min_ns_freq,
        'min_s_freq' : self.min_s_freq,
        'log_min_rel_freq': self.log_min_rel_freq,
        'HFW':  HFW,
        'num_patterns': self.num_patterns,
        'patterns': D}
        json.dump(data, file)
        
def load_PF(file):
    D = json.load(file)
    PF = PatternFeatures(set(D['HFW']), min_s_freq = 'min_s_freq', min_ns_freq = 'min_ns_freq')
    PF.log_min_rel_freq = D['log_min_rel_freq']
    PF.num_patterns = D['num_patterns']
    patterns = {}
    for m in D['patterns']:
        d = {}
        for words in D['patterns'][m]:
            d[tuple(words.split())] =  D['patterns'][m][words]
        patterns[m] = d
    PF.patterns = patterns
    return PF
    
    
class Pattern_data:
    def __init__(self, m, words, label, num, line, parent):
        self.count = [0,0]
        self.mask  = m
        self.count[label] = num
        self.words = words
        if not parent:
            self.original_instances = [line]
            self.original_occ = [num]
            self.parent_instances = []
            self.parent_occ = []
        else:
            self.original_instances = []
            self.original_occ = []
            self.parent_instances = [line]
            self.parent_occ = [num]
    def update(self, label, n, line, parent):
        self.count[label] += n
        if not parent:
            self.original_instances.append(line)
            self.original_occ.append(n)
        else:
            self.parent_instances.append(line)
            self.parent_occ.append(n)
    


def get_masks():
    masks = ['101','1101', '1011', '1001','11101','10111', '11011','10001','10101','11001','10011','1001001']
    reg = regex.compile(('|').join(masks))
    patterns = {}
    for m in masks:
        patterns[m] = {}
    return patterns, reg

def tokenize(comment):
    l = word_tokenize(comment.lower())
    is_word = [(regex.search(r"[^a-z0-9-']", w) == None)&(regex.search(r"[a-z0-9]", w) != None) for w in l]
    return l, is_word
    
def punctuation_features(comms):
    punct=['.', '!','?','"','(',',',')']
    reg = regex.compile('[' + (',').join(punct) + ']')
    data = {}
    i = 0
    for p in punct:
        data[p] = [[],[]]
    for c in comms:
        for p, count in Counter(reg.findall(c)).items():
            data[p][0].append(i)
            data[p][1].append(count/len(c))
        i = i+1
    num_comms = i
    v = []
    rows = []
    colns= []
    for i in range(0, len(punct)):
        v = v + data[punct[i]][1]
        rows = rows + data[punct[i]][0]
        colns = colns + [i]*len(data[punct[i]][0])
    return csr_matrix((v,(rows, colns)), shape = (num_comms, len(punct)))
    
def get_HFW(file_path, p = 1, min_freq = 1e-3):
    C = Counter()
    T = 0
    with open(file_path) as f:
        line = f.readline()
        while line:
            rand_draw = np.random.uniform()
            if rand_draw < p:
                split_line = line.split('\t')
                comms = [split_line[1], split_line[9]]
                for txt in comms:
                    tok, is_word = tokenize(txt)
                    T += sum(is_word)*min_freq
                    C.update([tok[i] for i in range(0, len(tok)) if is_word[i]])
            line = f.readline()
    
    return set([w for w, ct in C.items() if ct>T ])           
    
def parse_line(line):
    split_line = line.split('\t')
    return (int(split_line[0]), split_line[1], split_line[9])
    
def convert_to_df(M, PF, is_parent):
    if is_parent:
        tag = 'P'
    else:
        tag = 'O'
    cols = [0]*PF.num_patterns
    for m in PF.patterns:
        for w in PF.patterns[m]:
            cols[PF.patterns[m][w]] = (tag, m, w, PF.patterns[m][w])
    return pd.DataFrame(data = M, columns = cols)

def line_count(path):
    i = 0
    with open(path) as f:
        line = f.readline()
        while line:
            i = i+1
            line = f.readline()
    return i
            

training_path = '../data/samples/train-unbalanced.csv'
test_path = '../data/samples/test-unbalanced.csv'

print('Creating HFW\n')

HFW = get_HFW(training_path, p = 0.3)
PF = PatternFeatures(HFW)
print(HFW)

print('Fitting training data\n')
f = open(training_path)
lines = [parse_line(l) for l in f]
X_orig, X_parent  = PF.Fit_transform(lines, verbose = True)
f.close()
print('num. of features:{} \n'.format(PF.num_patterns))
f = open('../data/features/pattern_dict.json', 'w+')
PF.dump(f)
f.close()
save_npz('../data/features/pattern_training.npz', hstack([X_orig, X_parent]))

print('Transforming test data\n')
f = open(test_path)
lines = [parse_line(l)[1:3] for l in f]
Y_orig, Y_parent  = PF.Transform(lines)
f.close()
save_npz('../data/features/pattern_test.npz', hstack([Y_orig, Y_parent]))
