# Adam Maus & Brian Nixon
# CS761 Final Project
# Updated: 2012-04-27

# Libraries and Modules
from cae import CAE
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os.path
from helper_functions import *

# Path to the training file
# For my windows system path = C:\Users\melon\Desktop\mnist\mnist_train.amat
# os.path.expanduser(~melon) points to C:\Users\melon
# For linux, I think it will point to your home direcatory
training_file_name = os.path.expanduser('~melon') + "\Desktop\mnist\mnist_train.amat"
num_epochs = 100 # For 12,000 images, this takes ~20 minutes to run (if you remove the jacobi_loss term)
sample_size = 125 # set to None to use the entire training set
num_hidden_units = 1024
schatten_p_value = 1 # if you want to use infinity, use schatten_p_value = "inf"

# An image of the first training point alongside its reconstruction
# will be created and saved to this file
show_result = True
# If save_result_filename == None or "" then we don't save an image
save_result_filename = "result"
# Append the datetime to the save_result_filename
import datetime
d = datetime.datetime.now()
save_result_filename += "-" + str(d.year) + "-" + str(d.month) + "-" + str(d.day)
save_result_filename += "-" + str(d.hour) + str(d.minute)
save_result_filename += ".png"

# read_amat_file is in helper_functions
[X, Y] = read_amat_file(training_file_name, sample_size)

ae = CAE(epochs=num_epochs, n_hiddens=1024, schatten_p = schatten_p_value)
ae.fit(X, True)

r_X = ae.reconstruct(X[0])

# Show the first image and the reconstructed image
fig = plt.figure(1, (1,2))
grid = ImageGrid(fig, 111, nrows_ncols = (1, 2), axes_pad=0.1)
grid[0].imshow(numpy.reshape(X[0], (28,-1)))
grid[1].imshow(numpy.reshape(r_X, (28,-1)))

if save_result_filename != None and save_result_filename != "":
  plt.savefig(save_result_filename)
if show_result:
  plt.show()

