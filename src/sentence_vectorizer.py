import gzip
import re 
import string
import nltk
import time
import pickle
import math
from nltk.tokenize import word_tokenize
from numpy import zeros
from pymagnitude import Magnitude

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def preprocess(file_name, m, word2vec_magnitude_file):
        raw_data = open(file_name)
        X_pickle_file = open(file_name[:-4] + "_X.p",'wb') 
        y_pickle_file = open(file_name[:-4] + "_y.p",'wb') 
        comments = []
        y = []

        for line in raw_data: 
                y.append(int(line.split('\t')[0]))
                line = line.split('\t')[1]
                line = re.sub(r'\d+', '', line) 
                tokens = word_tokenize(line)
                comments.append(tokens)

        #max_len = len(max(comments,key=len)) #denote n
        max_len = 100
        # matrix m x n x d, m = sample, n = max_len of sentence, d = word vector size
        # TODO: If time allows, train on our data. For now, we just Google's pretrained model
        vector_size = 300
        #model = gensim.models.Word2Vec(comments, size=vector_size, window=8, min_count=1, workers=10)
        #model.train(comments, total_examples=len(comments), epochs=10)
        #TODO: Use FastText to handle unseen words, also Magnitude! 
        model = Magnitude(word2vec_magnitude_file)
        X = zeros([m,max_len,vector_size])
        #start = time.time()
        for comment_idx,sentence in enumerate(comments): 
                for sentence_idx, word in enumerate(sentence): 
                        if(sentence_idx == max_len -1): break
                        X[comment_idx][sentence_idx] = model.query(word)
                #if(comment_idx % 250 == 0):
                #       print(comment_idx)
                #      print(time_since(start))
        pickle.dump(X,X_pickle_file)
        pickle.dump(y,y_pickle_file)
        X_pickle_file.close()
        y_pickle_file.close()

num_samples = 764173 #lines in test 
filename = "./pol_test_cleaned.csv"
word2vec_magnitude_file = "./GoogleNews-vectors-negative300.magnitude"
preprocess(filename, num_samples, word2vec_magnitude_file)