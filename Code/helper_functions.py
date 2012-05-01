# .amat files have a digit stored in a vector on one line.
# the label is the last value in the vector
# if sample_size == None, then use the entire training set
import re, numpy
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

def pnorm(p, vect):
  if p == "inf":
    return max(vect)
  summ = 0
  for x in vect:
    summ += x**p
  return summ ** (1.0/float(p))
  
