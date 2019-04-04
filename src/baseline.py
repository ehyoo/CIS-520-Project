from sklearn import svm
import re
import time
import string
f = open("../data/train-balanced-sarc.csv", "r")
from collections import Counter
import nltk
import pandas as pd
import operator
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# text          = "i am awesome but bad i am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but badi am awesome but bad"

# The polarity score is a float within the range [-1.0, 1.0]
# where negative value indicates negative text and positive
# value indicates that the given text is positive.

# words = {'love'}
# nltk_words = list(stopwords.words('english'))
# grams = 2
#
#
# negative_senitments = []
# sarcastic_candid_counts = {}
# candid_counts = {}
# set_of_sentences_w_neg = {}
# for sentence in f:
#   split_sent = re.split(r'\t+', sentence)
#   label = split_sent[0]
#   comment = split_sent[1]
#   if label == "1" and words & set(comment.split()):
#     list_of_words = comment.split()
#     list_of_words = [w for w in list_of_words if not w in nltk_words]
#
#     for word in enumerate(words):
#       word = word[1]
#       if word in list_of_words:
#         next_words = list_of_words[list_of_words.index(word)+1:min(list_of_words.index(word) + 3, len(list_of_words)) ]
#         tags = [x[1] for x in nltk.pos_tag(next_words)]
#         if len(tags) > 0 and "VB" in tags[0]: #todo need to change
#
#            curr_word = ' '.join(next_words)
#            if len(curr_word.split()) < 2: #todo: need to change
#              print(curr_word)
#              negative_senitments.append(curr_word)
#              set_of_sentences_w_neg[curr_word] = 1
# positive_score_map = {}
# all_score_map = {}
# f = open("../data/train-balanced-sarc.csv", "r")
# for sentence in f:
#   split_sent = re.split(r'\t+', sentence)
#   label = split_sent[0]
#   comment = split_sent[1]
#   if set(negative_senitments)& set(comment.split()):
#     for neg_sent in negative_senitments:
#       split_sent = comment.split(neg_sent)
#       if len(split_sent) > 1 and len(split_sent[0].split()) > 1 :
#         preceding_word = split_sent[0].split()[-1]
#         words.add(preceding_word)
#         key = preceding_word.strip() + "#*$!#)!(#" + neg_sent.strip()
#         #increments scores
#         if label == "1":
#
#
#           if key in positive_score_map:
#             positive_score_map[key] += 1
#           else:
#             positive_score_map[key] = 1
#           if key in all_score_map:
#             all_score_map[key] += 1
#           else:
#             all_score_map[key] = 1
#
#         else:
#           if key in all_score_map:
#             all_score_map[key] += 1
#           else:
#             all_score_map[key] = 1
# print(positive_score_map)
# print(all_score_map)
# for key, value in all_score_map.iteritems():
#   if key not in positive_score_map:
#     all_score_map[key] = 0
#   else:
#     all_score_map[key] = positive_score_map[key]/ float(all_score_map[key])
# sorted_x = sorted(all_score_map.items(), key=operator.itemgetter(1))
# print(sorted_x)
#
# c = Counter(sorted_x)
# print(c.most_common(4))

sent_count = 0
count = lambda l1,l2: sum([1 for x in l1 if x in l2])
df = pd.DataFrame(columns=['count_puncutation', 'count_upper','count_emoticon', 'count_positive', 'count_negative', 'total_change','max_length'])

start_time = time.time()
rows_list = []
# import numpy
# dict1 = dict( (a,numpy.random.randint(100)) for a in ['A','B','C','D','E'])
# for i in range( 1,100-4):
#     dict1 = dict( (a,numpy.random.randint(100)) for a in ['A','B','C','D','E'])
#     rows_list.append(dict1)
# print(rows_list)
if True:
  x_feats = []
  y_feats = []
  for sentence in f:
    if sent_count < 10:
      split_sent = re.split(r'\t+', sentence)
      curr_comment = split_sent[1]
      label = split_sent[0]
      count_puncutation = count(curr_comment,set(string.punctuation))
      count_upper = sum(1 for c in curr_comment if c.isupper())
      count_emoticon = len(re.findall(r'\w+|[\U0001f600-\U0001f650]', curr_comment))
      count_positive = 0
      count_negative = 0

      split_sent = re.split(r'\s+', sentence)

      word_polarity = []
    # for word in split_sent:
    #   sent = TextBlob(word)
    #   curr_word_polarity = sent.sentiment.polarity
    #   if curr_word_polarity > 0:
    #     word_polarity.append(1)
    #   elif curr_word_polarity < 0:
    #     word_polarity.append(-1)
    #   else:
    #     word_polarity.append(0)

      sent_count +=1
      sentence_polarity = ['p' if TextBlob(x).sentiment.polarity > 0 else ('m' if TextBlob(x).sentiment.polarity < 0 else 'n') for x in sentence.split()]

      count_pos_then_neg = ''.join(sentence_polarity).count("mp")
      count_neg_then_pos = ''.join(sentence_polarity).count("pm")
      total_change = count_pos_then_neg + count_neg_then_pos
      longest_str_length = ''.join(sentence_polarity).split('n')
      max_length = max([len(s) for s in longest_str_length])
      curr_feat = [count_puncutation, count_upper,count_emoticon, count_positive, count_negative, total_change,max_length  ]
      dict1 = [{'qty1':10.0}]
      dict1 = {'count_puncutation':count_puncutation,'count_upper':count_upper,'count_emoticon':count_emoticon,
                'count_positive': count_positive, 'count_negative':count_negative, 'total_change':total_change,
                'max_length': max_length}
      rows_list.append(dict1)
      print(rows_list)


      # x_feats.append(curr_feat)
      if label == "1":
        y_feats.append(1)
      else:
        y_feats.append(0)
      # print(x_feats)
      # print(y_feats)
      if sent_count % 5000 == 0:
        print("--- %s seconds ---" % (time.time() - start_time))
        print(sent_count)
x_feats =  pd.DataFrame(rows_list, columns=['count_puncutation', 'count_upper','count_emoticon', 'count_positive', 'count_negative', 'total_change','max_length'])

print('done')
clf = svm.SVC(gamma='scale')
clf.fit(x_feats, y_feats)