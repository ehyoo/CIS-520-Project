from sklearn import svm
import re
import string
f = open("../data/train-balanced-sarc.csv", "r")
import nltk
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

words = {'love'}
nltk_words = list(stopwords.words('english'))
count = lambda l1,l2: sum([1 for x in l1 if x in l2])
sent_count = 0
grams = 2


negative_senitments = []
sarcastic_candid_counts = {}
candid_counts = {}
for sentence in f:
  split_sent = re.split(r'\t+', sentence)
  label = split_sent[0]
  comment = split_sent[1]
  if label == "1" and words & set(comment.split()):
    list_of_words = comment.split()
    list_of_words = [w for w in list_of_words if not w in nltk_words]

    for word in enumerate(words):
      word = word[1]
      if word in list_of_words:
        next_words = list_of_words[list_of_words.index(word)+1:min(list_of_words.index(word) + 3, len(list_of_words)) ]
        tags = [x[1] for x in nltk.pos_tag(next_words)]
        if len(tags) > 0 and "VB" in tags[0]:

           curr_word = ' '.join(next_words)
           negative_senitments.append(curr_word)
for sentence in f:
  split_sent = re.split(r'\t+', sentence)
  comment = split_sent[1]
  if label == "1" and words & set(comment.split()):
    list_of_words = comment.split()
    list_of_words = [w for w in list_of_words if not w in nltk_words]




if False:
  for sentence in f:
    if sent_count < -1:
      split_sent = re.split(r'\t+', sentence)
      curr_comment = split_sent[1]
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
    if sent_count % 1000 == 0:
      sentence_polarity = ['p' if TextBlob(x).sentiment.polarity > 0 else ('m' if TextBlob(x).sentiment.polarity < 0 else 'n') for x in sentence.split()]

      count_pos_then_neg = ''.join(sentence_polarity).count("mp")
      count_neg_then_pos = ''.join(sentence_polarity).count("pm")
      total_change = count_pos_then_neg + count_neg_then_pos
      longest_str_length = ''.join(sentence_polarity).split('n')
      max_length = max([len(s) for s in longest_str_length])



