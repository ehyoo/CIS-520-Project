# Import Libraries
from sklearn.model_selection import train_test_split
import gzip
import gensim 
import re 
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import zeros

# Data Processing
# All letter to lower case 
# Remove numbers
# removing punctuations
data_filename = "../tiny_sample.csv"
raw_data = open(data_filename)

# Sentence to lower case 
stop_words = set(stopwords.words('english'))
comments = []
labels = []

for line in raw_data: 
    labels.append(int(line.split('\t')[0]))
    line = line.split('\t')[1].lower()
    line = re.sub(r'\d+', '', line) 
    line = line.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(line)
    result = [i for i in tokens if not i in stop_words]
    comments.append(result)

max_len = len(max(comments,key=len)) #denote n
m = 10 #number of samples

# matrix m x n x d, m = sample, n = max_len of sentence, d = word vector size
# TODO: Save model? 
# build vocabulary and train model, TODO: change vector size to something reasonable 
vector_size = 10
model = gensim.models.Word2Vec(comments, size=vector_size, window=8, min_count=1, workers=10)
model.train(comments, total_examples=len(comments), epochs=10)

X = zeros([m,max_len,vector_size])
for comment_idx,sentence in enumerate(comments): 
    for sentence_idx, word in enumerate(sentence): 
        X[comment_idx][sentence_idx] = model.wv[word]

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(X, labels, test_size = 0.2, random_state = 42) 

#print(features_test)
#print(features_train)
#print(targets_test)
#print(targets_train)
