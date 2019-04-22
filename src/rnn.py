import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import random 
import time
import math 
import numpy as np
import pickle
from sentence_vectorizer import preprocess
from numpy import zeros
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.DoubleTensor(zeros((1,self.hidden_size))))

# create input + output tensors
# create init zero state
# read each word + keep hidden state for next word 

def train(category_tensor, line):
    rnn.zero_grad()
    hidden = rnn.init_hidden()

    for i in range(line.shape[0]):
        output, hidden = rnn(line_to_tensor([line[i]]), hidden)
    
    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data.item()

#convert list to variable 
def line_to_tensor(line): 
    return Variable(torch.from_numpy(np.array(line)))

def random_training_pair(x_train, y_train):      
    sentenceIndex = random.randint(0, x_train.shape[0] - 1)
    line = x_train[sentenceIndex]
    category = y_train[sentenceIndex]
    category_tensor = Variable(torch.LongTensor([category]))
    #line_tensor = line_to_tensor(line)
    return category, line, category_tensor  

def evaluate(line, model):
  hidden = model.init_hidden()

  for i in range(line.shape[0]):
      output, hidden = model(line_to_tensor([line[i]]), hidden)

  return output

'''
Input: trained model, a list of words, a list of class labels as integers
Output: a list of class labels as integers
'''
def predict(model, X, y):
  # iterate through each index in X convert to tensor with line to tensor
    output = []
    for sentence in X:
      pred = evaluate(sentence,model)
      _, top = pred.data.topk(1)
      output.append(top.item())
    return output

'''
Input: trained model, a list of words, a list of class labels as integers
Output: The accuracy of the given model on the given input X and target y

percentage labeled correctly
'''
def calculateAccuracy(model, X, y):
    prediction = predict(model, X, y)
    total = 0
    for i in range(len(prediction)):
      if prediction[i] == y[i]:
        total += 1
    return total / len(prediction)

'''
Returns the classification of the output from the rnn
Input: Output of the rnn (class probabilities)
Return: prediction
'''
def classifier_from_output(output):
    _ , top_i = output.topk(1)
    prediction = top_i[0].item()
    return prediction

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


X_training_data_filename = "/home/dkeren/Documents/Spring2019/CIS520/project/random_2000.pb"
y_training_data_filename = "/home/dkeren/Documents/Spring2019/CIS520/project/random_2000.pb"
X_testing_data_filename = "/home/dkeren/Documents/Spring2019/CIS520/project/random_1000.csv"
y_testing_data_filename = "/home/dkeren/Documents/Spring2019/CIS520/project/random_1000.csv"
word2vec_magnitude_file = "/home/dkeren/Documents/Spring2019/CIS520/project/GoogleNews-vectors-negative300.magnitude"

training_samples = 2000
testing_samples = 1000
#training_data = open(training_data_filename)
#testing_data = open(testing_data_filename)
x_train = pickle.load(open(X_training_data_filename))
y_train = pickle.load(open(y_training_data_filename))
x_test = pickle.load(open(X_testing_data_filename))
y_test = pickle.load(open(y_testing_data_filename))
#x_train, y_train = preprocess(training_data, training_samples,word2vec_magnitude_file)
#x_test, y_test = preprocess(testing_data,testing_samples,word2vec_magnitude_file)


learning_rate = 0.005
n_epochs = 1000
print_every = 50
plot_every = 50
vector_size = 300 #size of word2vec vector !!!!!!!!!!!!!!!!
n_categories = 2
torch.set_default_tensor_type(torch.DoubleTensor)
rnn = RNNModel(vector_size, 128, n_categories)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

current_loss = 0
loss_over_time = [] 
start = time.time()
print("starting training")
for epoch in range(1,n_epochs):
    # Get a random training input and target
    category, line, category_tensor = random_training_pair(x_train, y_train)
    output, loss = train(category_tensor, line)
    current_loss += loss
    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess = classifier_from_output(output)
        print(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f  / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, guess, correct))
    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        loss_over_time.append(current_loss / plot_every)
        current_loss = 0

print(calculateAccuracy(rnn, x_train, y_train))
plt.figure()
plt.plot(loss_over_time)
plt.ylabel("Loss")
plt.show()