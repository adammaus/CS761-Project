# Brian Nixon & Adam Maus
# CS761 - Final Project
# Updated: 2012-04-29
#
# This file takes a trained CAE and initializes a KNN classifier
#
# 1) Load a Numpy array file of a trained contractive autoencoder
#     - The file should contain the following Numpy arrays and vectors
#         W, c, b, CAE_params
# 2) Load the datasets (such as MNIST)
# 3) Measures the accuracy of the KNN on the test set
#
# Libraries and Modules
import os.path
import numpy
import random
from cae import CAE
from helper_functions import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import operator

# Parameters
numpy_array_filename = "results.npz"
training_file_name = os.path.expanduser('~melon') + "\Desktop\mnist\mnist_train.amat"
testing_file_name = os.path.expanduser('~melon') + "\Desktop\mnist\mnist_test.amat"
training_sample_size = 1000 # MNIST has ~12,000 samples
testing_sample_size = 100   # MNIST has ~50,000 samples
k = 10 # Number of neighbors to use
p = 2  # P norm value to use

# 1) Load a Numpy array file of a trained contractive autoencoder
data = numpy.load(numpy_array_filename)
W = data['W']
c = data['c']
b = data['b']
CAE_params = data['CAE_params']
[n_hidden,learning_rate,jacobi_penalty,batch_size,epochs,schatten_p, loss] = CAE_params
X = data['X'] # The examples used during cae.fit

# Load the data into a CAE
ae = CAE( n_hiddens=n_hidden,
          W=W,
          c=c,
          b=b,
          learning_rate=learning_rate,
          jacobi_penalty=jacobi_penalty,
          batch_size=batch_size,
          epochs=epochs,
          schatten_p=schatten_p )

# 2) Load the datasets (such as MNIST)
# read_amat_file is in helper_functions
[rX, rY] = read_amat_file(training_file_name, training_sample_size)
[tX, tY] = read_amat_file(testing_file_name, testing_sample_size)

# For each training point, encode
encoded_rX = []
for x in rX:
  encoded_rX.append(ae.encode(x))

# 3) Measures the accuracy of the KNN on the test set
# For each testing point, encode and see how it compares
correct = 0
incorrect = 0
total = 0
j = 0
while j < len(tX):
  x = tX[j]
  encoded_x = ae.encode(x)
  # Find the closest training point
  distances = {}
  i = 0
  while i < len(encoded_rX):
    distances[i] = pnorm(p, (encoded_rX[i] - encoded_x))
    i = i + 1
  # Sort the distances be closeness
  sorted_dists = sorted(distances.iteritems(), key=operator.itemgetter(1))
  # Choose the top k values and have them vote for the value
  votes = {}
  most_votes_count = 0
  most_votes_label = 0
  for y in sorted_dists[0:k]:
    label = rY[y[0]]
    if not label in votes:
      votes[label] = 0
    votes[label] += 1
    if votes[label] > most_votes_count:
      most_votes_count = votes[label]
      most_votes_label = label
    #print y[0], y[1], rY[y[0]]

  # Compare the most votes to the actual label
  if tY[j] == most_votes_label:
    correct += 1
  else:
    incorrect += 1
  total += 1
  j = j + 1
  print correct, incorrect
