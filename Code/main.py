from cae import CAE
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os.path

# Adam Maus & Brian Nixon
# CS761 Final Project
# Updated: 2012-04-27

# Path to the training file
# For my windows system path = C:\Users\melon\Desktop\mnist\mnist_train.amat
# os.path.expanduser(~melon) points to C:\Users\melon
# For linux, I think it will point to your home direcatory
training_file_name = os.path.expanduser('~melon') + "\Desktop\mnist\mnist_train.amat"
num_epochs = 100 # For 12,000 images, this takes ~20 minutes to run (if you remove the jacobi_loss term)
sample_size = 10 # set to None to use the entire training set
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

# .amat files have a digit stored in a vector on one line.
# the label is the last value in the vector
# if sample_size == None, then use the entire training set
import re
def read_amat_file(file_name, sample_size=None):
  f = open(file_name, "r")
  temp_arr = f.readlines()
  f.close()
  if sample_size == None:
    sample_size = len(temp_arr)-1
  
  X = []
  Y = []
  for x in temp_arr[0:sample_size]:
    # Split each line and then convert to floating numbers
    x = x.strip()
    x_str_arr = re.split('\s+', x)
    if len(x_str_arr) > 0:
      x_flo_arr = []
      for feature in x_str_arr:
        x_flo_arr.append(float(feature))
      X.append(x_flo_arr[0:len(x_flo_arr)-1])
      Y.append(x_flo_arr[len(x_flo_arr)-1])
  del temp_arr # Force garbage collection
  return [numpy.array(X), numpy.array(Y)]


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

