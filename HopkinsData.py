import pandas as pd
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as feature_column_lib
tf.reset_default_graph()
import numpy as np
from hTools import XLS2df
from hDataframe import batch, shuffle, CSV2df



def normalize(tensor):
  tf.div(
   tf.subtract(
      tensor, 
      tf.reduce_min(tensor)
   ), 
   tf.subtract(
      tf.reduce_max(tensor), 
      tf.reduce_min(tensor)
   )
  )
  return tensor


def preprocess(df):
  #Only use numerical data
  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  df = df.select_dtypes(include=numerics)
  df = df.astype('float32')    
  #replace all 9999s with NaN
  df = df.mask(df == 9999.0, np.nan)

    #normalize by as appropriate (ignore NaN)
  df = (df-df.min())/(df.max()-df.min())
  #replace all NaNs with the average value
  df = df.fillna(0.5) 

  #select the fields to include
  df = df.filter(items=[
#      'BF_kHz', 
#      'depth_from_surface',
      'mean_spont',
      'Th_dB',
#      'tone_noise_ratio',
      'max_rate',
#      'frequency',
#      'stim_level',
#      'SLI',
#      'E',
#      'LLI',
#      'LLE',
#      'LTE_I',
#      'LTE_II',
#      'LTE_III',
#      'FSL_on',
#      'FSL_on_std',
#      'FSL_off',
#      'FSL_off_std',
#      'FISI_on',
#      'FISI_on_std',
#      'FISI_off',
#      'FISI_off_std',
      'driven_on',
#      'driven_on_std',
#      'driven_on_fano',
      'driven_off',
#      'driven_off_std',
#      'driven_off_fano',
      'driven_diff',
#      'total_on',
#      'total_off',
      'total_diff',
      '250-350_on',
      '250-350_on_std',
      '250-350_off',
      '250-350_off_std',
      '250-350_d_prime',
      '350-450_on',
      '350-450_on_std',
      '350-450_off',
      '350-450_off_std',
      '350-450_d_prime',
      '450-550_on',
      '450-550_on_std',
      '450-550_off',
      '450-550_off_std',
      '450-550_d_prime',
      '300-400_on',
      '300-400_on_std',
      '300-400_off',
      '300-400_off_std',
      '300-400_d_prime',
      '400-500_on',
      '400-500_on_std',
      '400-500_off',
      '400-500_off_std',
      '400-500_d_prime',
      '500-1000_on',
      '500-1000_off',
      '500-1000_d_prime',
#      'level_dB_SPL',
#      'ISI_on_mean',
#      'ISI_off_mean',
#      'ISI_on_std',
#      'ISI_off_std',
#      'ISI_on_Cv',
#      'ISI_off_Cv',
#      'ISI_on_fano',
#      'ISI_off_fano',
#      'ISI_on_max',
#      'ISI_off_max',
#      'ISI_on_mode',
#      'ISI_off_mode',
#      'ISI_on_mode_duration',
#      'ISI_off_mode_duration',
#      'ISI_on_q',
#      'ISI_off_q',
#      'ISI_on_skewness',
#      'ISI_off_skewness',
#      'LTE_II_saturation',
#      'LTE_rate_on',
#      'LTE_rate_off',
      'LTE_II_d_prime',
      'd_prime_on',
      'd_prime_off',
      'd_prime_on_std',
      'd_prime_off_std',
      'd_prime',
      'class'
      ])  
      
  #normalize by as appropriate (ignore NaN)
  df = (df-df.min())/(df.max()-df.min())
  #replace all NaNs with the average value
  df = df.fillna(0.5) 

  #split into train and test dataframes
  df = shuffle(df)   
  testDF, trainDF = df[:150], df[150:]
  return trainDF, testDF 
 
       
       

def convert_to_one_hot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)
    result = np.zeros((len(vector), num_classes))
    result[np.arange(len(vector)), vector.flatten()] = 1
    return result.astype(int)
    
    
def train_input_fn(df, batch_size=150):
    """An input function for training"""
    fts = df.drop(columns=['class'])
    labs = df.filter(items=['class']).values.astype(int)

    features = {k:list(v.values) for k,v in fts.items()}
    features = dict(features)
    x = fts.values
    x = np.array([[x]]).reshape((np.shape(x)[0], np.shape(x)[1], 1, 1))
#    x = tf.feature_column.input_layer(features, define_fc(),weight_collections="input_weights", trainable=True)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices({"x_ph":x,"y_ph":convert_to_one_hot(labs)})
    
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).batch(batch_size).repeat()
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
    

def eval_input_fn(df):
    """An input function for evaluation"""
    fts = df.drop(columns=['class'])
    labs = df.filter(items=['class']).values.astype(int)

    features = {k:list(v.values) for k,v in fts.items()}
    features = dict(features)
    x = fts.values
    x = np.array([[x]]).reshape((np.shape(x)[0], np.shape(x)[1], 1, 1))
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices({"x_ph":x,"y_ph":convert_to_one_hot(labs)})
    
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).batch(np.shape(x)[0]).repeat()
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


