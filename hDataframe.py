import pandas as pd
import hDictionary
  
#############################################
def shuffle(df):
  ###########################################
  # This function will shuffle the rows
  df = df.sample(frac=1).reset_index(drop=True)  
  return df
#############################################


#############################################
def batch(df, n):
  ###########################################
  # This function will return a batch of data
  batch, therest = df[:n], df[n:]
  return batch, therest
#############################################


#############################################
def XLS2df(filename, sheet):
  ###########################################
  # This function will import the specified sheet from the *.xls file, 
  # split the data into train and test sets
  # and return them as dataframes
  df = pd.read_excel(filename, sheet_name=sheet)
  return df
#############################################


#############################################
def CSV2df(filename):
  ###########################################

  df = pd.read_csv(filename, sep=',')
  return df
#############################################

