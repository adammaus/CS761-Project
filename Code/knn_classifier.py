# Brian Nixon & Adam Maus
# CS761 - Final Project
# Updated: 2012-04-30
#
# This file initializes a KNN classifier and runs it on a data set
#
# 1) Load the datasets (such as MNIST)
# 2) Measures the accuracy of the KNN on the test set
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
training_file_name = os.path.expanduser('~melon') + "\Desktop\mnist\mnist_train.amat"
testing_file_name = os.path.expanduser('~melon') + "\Desktop\mnist\mnist_test.amat"
training_sample_size = 400 # MNIST has ~12,000 samples
testing_sample_size = 400   # MNIST has ~50,000 samples
k = 20 # Number of neighbors to use
p = 2  # P norm value to use

# 1) Load the datasets (such as MNIST)
# read_amat_file is in helper_functions
[rX, rY] = read_amat_file(training_file_name, training_sample_size)
[tX, tY] = read_amat_file(testing_file_name, testing_sample_size)

# 2) Measures the accuracy of the KNN on the test set
# For each testing point, encode and see how it compares
correct = 0
incorrect = 0
total = 0
j = 0
while j < len(tX):
  x = tX[j]
  # Find the closest training point
  distances = {}
  i = 0
  while i < len(rX):
    distances[i] = pnorm(p, (rX[i] - x))
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
  # Sort the votes
  sorted_votes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
  labels = []
  most_votes_count = -1
  for y in sorted_votes:
    if most_votes_count == -1 or most_votes_count == y[1]:
      labels.append(y[0])
      most_votes_count = y[1]
  most_votes_label = labels[random.randint(0, len(labels)-1)]  
  # Compare the most votes to the actual label
  if tY[j] == most_votes_label:
    correct += 1
  else:
    incorrect += 1
  total += 1
  j = j + 1
  if total % 100 == 0:
    print correct, incorrect
print correct, incorrect
