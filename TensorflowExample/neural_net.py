""" Code that implements a basic fully-connected neural network 
    in Tensorflow. 

    Neural networks have many parameters that can be tuned. You can 
    tweak the parameters when running this code by feeding them 
    into the class constructor, or simply through command line 
    arguments. For example:
        python neural_net.py --learning_rate .001
    sets the learning rate to .001. To see all of the parameters 
    that can be tweaked via the command line, scroll down to the 
    __main__ function.

    Tensorflow is an efficient framework for training neural 
    networks. Rather than running the code purely in python, it 
    does most of its computational in C. In order to do this, it 
    requires that you set up a pre-defined computation Graph, 
    which contains tensors - variables that do not necessarily 
    store actual values until they are evaluated. The graph 
    determines the flow of information through the neural network. 
    Backpropagation is automatically performed for all variables 
    in the graph that contribute to computing the outcome that we 
    are trying to optimize. When we are ready to train the model,
    we initialize a Session and start feeding data into the Graph. 
    You can read more about how Tensorflow works at 
    https://www.tensorflow.org/versions/r0.11/get_started/basic_usage

    This code implements simple fully-connected network layers. To
    learn how to build a convolutional network, please see:
    https://www.tensorflow.org/tutorials/deep_cnn
    To learn about recurrent neural networks, see:
    https://www.tensorflow.org/tutorials/recurrent
"""

import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import sys
import math
import time
import argparse

DEFAULT_DATA_FILE = 'art_data.pickle'

class NeuralNetwork:
    def __init__(self, filename, layer_sizes=[128,64], batch_size=10, 
                 learning_rate=.01, dropout_prob=1.0, weight_penalty=0.0):
        '''Initialize the class by loading the required datasets 
        and building the graph.

        Args:
            filename: a file containing the data.'''

        # Extract the data from the filename
        #self.input_size

        # Hyperparameters that should be tuned
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob # set to < 1.0 to apply dropout, 1.0 to remove
        self.weight_penalty = weight_penalty # set to > 0.0 to apply weight penalty, 0.0 to remove

        # Hyperparameters that could be tuned 
        # (but are probably the best to use)
        self.activation_func = 'relu'
        self.optimizer = tf.train.AdamOptimizer

        # Set up tensorflow computation graph.
        self.graph = tf.Graph()
        self.build_graph()

    def initialize_network_weights(self):
        """Constructs Tensorflow variables for the weights and biases
        in each layer of the graph. These variables will be updated
        as the network learns.

        The number of layers and the sizes of each layer are defined
        in the class's layer_sizes field.
        """
		sizes = []
		self.weights = []
		self.biases = []
		for i in range(len(self.layer_sizes)+1):
			if i==0:
				input_len = self.input_size # X second dimension
			else:
				input_len = self.layer_sizes[i-1]
			
			if i==len(self.layer_sizes):
				output_len = self.embedding_size
			else:
				output_len = self.layer_sizes[i]
				
			layer_weights = weight_variable([input_len, output_len],name='weights' + str(i))
			layer_biases = bias_variable([output_len], name='biases' + str(i))
			
			self.weights.append(layer_weights)
			self.biases.append(layer_biases)
			sizes.append((str(input_len) + "x" + str(output_len), str(output_len)))
		
		if self.verbose:
			print("Okay, making a neural net with the following structure:")
			print(sizes)

    def build_graph(self):
        """Constructs the tensorflow computation graph containing all variables
        that will be trained."""
        print '\nBuilding computation graph...'

        with self.graph.as_default():
            # Placeholders can be used to feed in different data during training time.
            self.tf_X = tf.placeholder(tf.float64, name="X") # features
		    self.tf_Y = tf.placeholder(tf.float64, name="Y") # labels
            self.dropout_keep_prob = tf.placeholder(tf.float32) # Implements dropout

            # Place the network weights/parameters that will be learned into the 
            # computation graph.
            self.initialize_network_weights()

            # Defines the actual network computations using the weights. 
            def run_network(input_X):
                hidden = input_X
                for i in range(len(self.weights)):
                    with tf.name_scope('layer' + str(i)) as scope:
                        # tf.matmul is a simple fully connected layer. 
                        hidden = tf.matmul(hidden, self.weights[i]) + self.biases[i]
                        
                        if i < len(self.weights)-1:
                            # Apply activation function
                            if self.activation_func == 'relu'
                                hidden = tf.nn.relu(hidden) 
                            else:
                                raise ValueError('That activation function has not been implemented.')

                            # Apply dropout
                            hidden = tf.nn.dropout(hidden, self.dropout_keep_prob) 
                return hidden
            self.run_network = run_network

            # Compute the loss function
            self.logits = run_network(self.tf_X)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_Y))

            # Add weight decay regularization term to loss
            self.loss += self.weight_penalty * sum([tf.nn.l2_loss(w) for w in self.weights])

            # Set up backpropagation computation!
            self.opt_step = self.optimizer(self.learning_rate).minimize(self.loss)

            # Predicting a new point
            self.Y_hat = tf.nn.softmax(self.logits)

            self.init = tf.initialize_all_variables()

    def load_pickled_dataset(self, pickle_file):
        print "Loading datasets..."
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            self.train_X = save['train_data']
            self.train_Y = save['train_labels']
            self.val_X = save['val_data']
            self.val_Y = save['val_labels']

            if INCLUDE_TEST_SET:
                self.test_X = save['test_data']
                self.test_Y = save['test_labels']
            del save  # hint to help gc free up memory
        print 'Training set', self.train_X.shape, self.train_Y.shape
        print 'Validation set', self.val_X.shape, self.val_Y.shape
        if INCLUDE_TEST_SET: print 'Test set', self.test_X.shape, self.test_Y.shape



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

if __name__ == '__main__':
    invariance = False
    if len(sys.argv) > 1 and sys.argv[1] == 'invariance':
        print "Testing finished model on invariance datasets!"
        invariance = True

    parser = argparse.ArgumentParser(description='Open and query encrypted SQL files')
    parser.add_argument('-k', '--key', dest='key', required=True,
                        help='Private key to decrypt the files')
    args = parser.parse_args()
    
    t1 = time.time()
    conv_net = ArtistConvNet(invariance=invariance)
    conv_net.train_model()
    t2 = time.time()
    print "Finished training. Total time taken:", t2-t1
