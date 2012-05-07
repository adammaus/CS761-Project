# Adam Maus & Brian Nixon
# CS761 Final Project
# Updated: 2012-04-27
#
# This program measures the variation in encoding while varying a training
# point's features.
#
# 1) Loads a Numpy array file of a trained contractive autoencoder
#     - The file should contain the following Numpy arrays and vectors
#         W, c, b, CAE_params
# 2) Pull a dataset (such as MNIST) and use a single data point
# 3) Measure the difference between encoding the data point and variations
#    in the data point's features

# Libraries and Modules
import os.path
import numpy
import random
from cae import CAE
from helper_functions import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Parameters
numpy_array_filename = "result-1-1024-10000-2012-5-1-1447.npz"
training_file_name = "\mnist\mnist_train.amat"

# Read the numpy array file
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

# Pick a random point in the data
pt_index = 0 #random.randint(0, len(X)-1)
pt = X[pt_index]


## Output the pt in a graph and its reconstruction by the CAE
#target = numpy.reshape(pt, (28,-1))
#reconstruction = numpy.reshape(ae.reconstruct(pt), (28,-1))
#num_cols = 4
#fig = plt.figure(2, (1,num_cols))
#grid = ImageGrid(fig, 111, nrows_ncols = (2, num_cols), axes_pad=0.1)
#[U, S, V] = numpy.linalg.svd(ae.jacobian(X)[pt_index])
#num = num_cols * 0
#grid[num + 0].imshow(target)
#grid[num + 1].imshow(reconstruction)
#grid[num + 2].imshow(numpy.reshape(V[0], (28,-1)))
#grid[num + 3].imshow(numpy.reshape(V[1], (28,-1)))
#num = num_cols * 1
#eps = 0.01
#reconstruction = numpy.reshape(ae.reconstruct(pt + eps), (28,-1))
#grid[num + 0].imshow(reconstruction)
#eps = -0.01
#reconstruction = numpy.reshape(ae.reconstruct(pt + eps), (28,-1))
#grid[num + 1].imshow(reconstruction)
#eps = 0.05
#reconstruction = numpy.reshape(ae.reconstruct(pt + eps), (28,-1))
#grid[num + 2].imshow(reconstruction)
#eps = -0.05
#reconstruction = numpy.reshape(ae.reconstruct(pt + eps), (28,-1))
#grid[num + 3].imshow(reconstruction)
#plt.show()


# Show the contraction by putting a sphere of radius eps around the target
# and measure the variation in encoding between the target and new point
print "Epsilon, Variation in Encoding"
encoded = ae.encode(pt)
vnum = 10
v = []
for j in range(0, vnum):
  vdir = [random.uniform(-1,1) for i in range(0, len(pt))]
  vlen = pnorm(2, vdir)
  v.append([i/vlen for i in vdir])
for h in range(30, 0, -2):
  h_contr = 0
  for j in range(0, vnum):
    encoded_v = ae.encode([i*h for i in v[j]])
    h_contr += pnorm(2, encoded-encoded_v)/h
  print h, h_contr/vnum

#eps = -0.1
#while eps >= 0.1:
#  temp_pt = []
#  j = 0
#  while j < len(pt):
#    if pt[j] != 0 or True:
#      temp_pt.append(pt[j] + eps)
#    else:
#      temp_pt.append(pt[j])
#    j = j + 1
#  encoded_eps = ae.encode(temp_pt)
#  print eps,"\t,\t", pnorm(2, (encoded - encoded_eps))
#  eps += 0.001

