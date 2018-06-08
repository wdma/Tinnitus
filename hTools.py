import pandas as pd
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from hDataframe import shuffle


#############################################
def graph(sess):
  ###########################################
  # This function starts the Tensorflow graphing function
  writer = tf.summary.FileWriter('./graph', sess.graph)
  return writer
#############################################


#############################################
def XLS2df(filename, sheet):
  ###########################################
  # This function will import the specified sheet from the *.xls file, 
  # split the data into train and test sets
  # and return them as dataframes
  df = pd.read_excel(filename, sheet_name=sheet)
  #shuffle the rows for good measure
  df = shuffle(df)
  
  return df
#############################################


#############################################
def console(output, message):
  ###########################################
    print_statement = tf.Print(output, [output], message = message)
    return print_statement
