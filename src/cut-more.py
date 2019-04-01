# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:32:09 2019

@author: artur
"""
import numpy as np 

# First, set our seed.
np.random.seed(14)

limit = 100030
counter = 1

CUTOFF = 0.03

subreddits_dict = {}

line_cache = []
label_cache = []
score_cache = []
ups_cache = []
downs_cache = []
comment_wc_cache = []
parent_wc_cache = []
total_sarc = 0
cut_sarc = 0
cut_line_count = 0

path_org = '../data/samples/sample_train_unbalanced.csv'
path_target = '../data/samples/sample_train_unbalanced_red.csv'

def get_word_count_of_comment(comment):
    return str(len(comment.split())) # Simple word count by splitting on space.

with open(path_org) as f:
    while True:
        line = f.readline()
        if not line:
            print("Reached end.") # Just to keep track of where we are.
            with open(path_target, 'a') as note:
                for selected_line in line_cache:
                    note.write(selected_line)
            print("no more lines")
            print("total commets: {:d} \n total sarcatis comments: {:d} \n perc: {:f}".format(counter, total_sarc, 100*total_sarc/counter))
            print("Reduced commets: {:d} \n reduces sarcatis comments: {:d} \n perc: {:f}".format(cut_line_count, cut_sarc,100*cut_sarc/cut_line_count))
            break

        split_line = line.split('\t')
        # Firstly, getting our label
        label = split_line[0]
        label_cache.append(label)
        total_sarc = total_sarc + int(label)
        # Then, the subreddit
        subreddit = split_line[3]
        # Then, the score, ups, and downs
        score = split_line[4]
        ups = split_line[5]
        downs = split_line[6]
        score_cache.append(score)
        ups_cache.append(ups)
        downs_cache.append(downs)

        # Then, the Parent and Child word count
        parent_comment = split_line[9]
        original_comment = split_line[1]
        parent_comment_wc = get_word_count_of_comment(parent_comment)
        original_comment_wc = get_word_count_of_comment(original_comment)
        parent_wc_cache.append(parent_comment_wc)
        comment_wc_cache.append(original_comment_wc)

        # Then, get our sample
        rand_draw = np.random.uniform()
        if rand_draw <= CUTOFF:
            line_cache.append(line)
            cut_line_count = cut_line_count +1
            cut_sarc = cut_sarc + int(label)

        if counter % 1000000 == 0:
            print("Reached " + str(counter) + "... Writing results to disk") # Just to keep track of where we are.
            with open(path_target, 'a') as note:
                for selected_line in line_cache:
                    note.write(selected_line)

            line_cache = []
            label_cache = []
            score_cache = []
            ups_cache = []
            downs_cache = []
            comment_wc_cache = []
            parent_wc_cache = []

        counter = counter + 1
