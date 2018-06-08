# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import os.path
import shutil
import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.tensorboard.plugins import projector
from hDataframe import CSV2df, batch, shuffle
from hTools import graph, console
from HopkinsData import preprocess, convert_to_one_hot, train_input_fn, eval_input_fn


#DATA = "./test.csv"
DATA = "./HopkinsData.csv"
#LABELS = "./HopkinsDataLabels.tsv"


class Config:
    num_classes = 2                 # number of classes in data
    batchsize = 150                 # need to figure how best to apply this   
    learning_rate = [1E-3, 1E-4]   # Gradient descent learning rates.
    num_epochs = 5001              # Gradient descent number of iterations.


class Data:
    """
    Utility class for loading training and test CSV files.
    """
    def __init__(self):
        self.training_set = None
        self.test_set = None
        
    def load(self, DATA, num_classes=None):
        """
        Load CSV files into class member variables.
        """
        df = CSV2df(DATA)
        training_set, test_set = preprocess(df)
        
        self.training_set = training_set
        self.test_set = test_set


class Classifier:
    """
    Trains a dense neural network model.
    """
    def __init__(self):
        self.data = None


    def loadData(self):
        """
        Load data from CSV files.
        """
        self.data = Data()
        self.data.load(DATA, Config.num_classes)
        
    
    def visualize_embeddings(self, sess, tensor, name):
        """
        Visualises an embedding vector into Tensorboard

        :param sess: Tensorflow session object
        :param tensor:  The embedding tensor to be visualizd
        :param name: Name of the tensor
        """
        # make directory if not exist
        if not tf.os.path.exists(self.save_dir):
            tf.os.makedirs(self.save_dir)
        # summary writer
        summary_writer = tf.summary.FileWriter(self.save_dir, graph=tf.get_default_graph())
        # embedding visualizer
        config = projector.ProjectorConfig()
        emb = config.embeddings.add()
        emb.tensor_name = name  # tensor
        emb.metadata_path = tf.os.path.join(self.save_dir, self.meta_file)  # metadata file
        print(tf.os.path.abspath(emb.metadata_path))
        projector.visualize_embeddings(summary_writer, config) 
        
    ####################
    #Layers
    ####################
    def conv_layer(self, input, size_in, size_out, name="conv"):
      with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([1, self.data.training_set.shape[1]-1, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        sub = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding="SAME")
        pooled = tf.nn.max_pool(sub, ksize=[1, 2, 1, 1], strides=[1, 1, 2, 1], padding="SAME")
        act = tf.nn.relu(pooled)
        
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)        
        
        return act


    def fc_layer(self, input, size_in, size_out, name="fc"):
      with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        sub = tf.matmul(input, w) + b
        act = tf.nn.relu(sub)
        
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        
        return act
        
    def output_layer(self, input, name="prediction"):
      with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([1,2], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[2]),name="B")
        sub = (w * input) + b
        y_hat = tf.nn.softmax(sub)
        
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", y_hat)
        
        return y_hat
            
    ####################
    #The model
    ####################
    def model(self, learning_rate, nfc, nconv, hparam):
      tf.reset_default_graph()
      num_features = self.data.training_set.shape[1]-1
      num_classes = Config.num_classes
      
      with tf.name_scope('input'):
        x_ph = tf.placeholder(tf.float32, [None, num_features,1,1], name="X")
        y_ph = tf.placeholder(tf.float32, [None, num_classes], name="Y")

        trainer = train_input_fn(self.data.training_set, Config.batchsize)
        tester = eval_input_fn(self.data.test_set)
      ####################
      #Assembly
      ####################
      if nconv==2:
        act = self.conv_layer(x_ph, 1, 32, "convolution")
        act = self.conv_layer(act, 32, 64, "convolution")
      elif nconv==1:
        act = self.conv_layer(x_ph, 1, 64, "convolution")
      else:
        print("hi")
        #need to fill this part
        
      flattened = tf.reshape(act, [-1, num_features * 64])

      if nfc==2:
        act = self.fc_layer(flattened, num_features * 64, 1024, "matrixMultiply")
        embedding_input = act
        embedding_size = 1024
        tf.summary.histogram("fc1", act)
        
        act = self.fc_layer(act, 1024, 1, "matrixMultiply")  
      elif nfc==1:
        act = self.fc_layer(flattened, num_features * 64, 1, "matrixMultiply")
        embedding_input = flattened
        embedding_size = num_features * 64
        tf.summary.histogram("fc1", flattened)
      else:
        print("hi")
        #need to fill this part
      
      y_hat = self.output_layer(act,"output")    

      ####################
      #Operations
      ####################             
      with tf.name_scope("loss"):
        loss = tf.losses.mean_squared_error(predictions=y_hat, labels=y_ph)  
        tapLoss = console(loss,"tap")
        tf.summary.scalar("error", loss)
        tapGroundtruth = console(y_ph,"tap")

      with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
      
      with tf.name_scope('score'):
        correct_prediction = tf.equal(tf.arg_max(y_hat,1), tf.arg_max(y_ph,1)) # List of T,F
        tapScore = console(correct_prediction,"tap")
        
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
      
      
      ####################
      #Runtime
      ####################  
      summ = tf.summary.merge_all()
      saver = tf.train.Saver()
      
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        writer = tf.summary.FileWriter("./graph/" + hparam)
        writer.add_graph(sess.graph)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        #embedding.tensor_name = "test_embedding"
        #embedding.metadata_path = "HopkinsDataLabels.tsv"
        projector.visualize_embeddings(writer, config)
        
        for i in range(Config.num_epochs):
          feed = sess.run(trainer)
          _, loss_out = sess.run([train_step, loss], feed_dict={x_ph: feed['x_ph'], y_ph: feed['y_ph']})
          if (i % 1000 == 0):
            s = sess.run(summ, feed_dict={x_ph: feed['x_ph'], y_ph: feed['y_ph']})
            writer.add_summary(s, i)
            print("Epoch %6d/%6d: Loss=%10.5f" % (i, Config.num_epochs, loss_out) )
            saver.save(sess, "./graph/"+ hparam , i)
        writer.close()
    
        # Compute accuracy on test set.
        feed = sess.run(tester)
        correct_predictions, accuracy = \
          sess.run([correct_prediction, accuracy], feed_dict={x_ph: feed['x_ph'], y_ph: feed['y_ph']})
        
        print()
        print("Predictions on test data:")
        print(correct_predictions)
        print("Test accuracy = %.3f" % accuracy)

 

def make_hparam_string(learning_rate, nfc, nconv):
  if nconv==2:
    conv_param = "conv=2"
  elif nconv==1:
    conv_param = "conv=1"
  else: conv_param = "conv=0"
  
  if nfc==2:
    fc_param = "fc=2" 
  elif nfc==1:
    fc_param = "fc=1" 
  else: fc_param = "fc=0"
  return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)


def main():
  classifier = Classifier()
  classifier.loadData()
  # You can try adding some more learning rates
  for learning_rate in Config.learning_rate:
    # Include "False" as a value to try different model architectures
    for nfc in [1, 2]:
      for nconv in [1, 2]:
        # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2")
        hparam = make_hparam_string(learning_rate, nfc, nconv)
        print('Starting run for %s' % hparam)
        # Actually run with the new settings
        classifier.model(learning_rate, nfc, nconv, hparam)
  print('Done training!')
  print('Run `tensorboard --logdir=./graph` to see the results.')


if __name__ == '__main__':
  main()
