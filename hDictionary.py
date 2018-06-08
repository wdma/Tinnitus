import pandas as pd
import tensorflow as tf


#############################################
def pull(dictionary, field):
  ###########################################
  #this function will return all of the values in a give field
  data = list(dictionary[field].values())
  return data
#############################################


#############################################
def getFlags(dictionary, field):
  ###########################################
  # This function will return the specified field as flags and the rest of the dictionary
  flags = dictionary.pop(field)
  
  return dictionary, flags
#############################################

#############################################
def mapdict(f,d):
  ###########################################
  # This function maps a function across a dictionary
  for k, v in d.iteritems():
    d[k] = f(v)
#my_dictionary = {'a':1, 'b':2}
#mutate_dict(lambda x: x+1, my_dictionary)
#############################################

