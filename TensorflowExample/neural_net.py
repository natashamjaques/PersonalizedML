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
import matplotlib.pyplot as plt
import sys
import math
import time
import argparse

import data_funcs

DEFAULT_DATA_FILE = 'art_data.pickle'

def reload_files():
    reload(data_funcs)

class NeuralNetwork:
    def __init__(self, filename, model_name, layer_sizes=[128,64], batch_size=10, 
                 learning_rate=.01, dropout_prob=1.0, weight_penalty=0.0,
                 output_type='classification', checkpoint_dir='./saved_models/'):
        '''Initialize the class by loading the required datasets 
        and building the graph.

        Args:
            filename: a file containing the data.
            model_name: name of the model being trained. Used in saving
                model checkpoints.
            layer_sizes: a list of sizes of the neural network layers.
            batch_size: number of training examples in each training batch. 
            learning_rate: the initial learning rate used in stochastic 
                gradient descent.
            dropout_prob: the probability that a node in the network will not
                be dropped out during training. Set to < 1.0 to apply dropout, 
                1.0 to remove dropout.
            weight_penalty: the coefficient of the L2 weight regularization
                applied to the loss function. Set to > 0.0 to apply weight 
                regularization, 0.0 to remove.
            output_type: the type of output prediction. Either 'classification'
                or 'regression'.
            checkpoint_dir: the directly where the model will save checkpoints,
                saved files containing trained network weights.
            '''
        # Hyperparameters that should be tuned
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob 
        self.weight_penalty = weight_penalty 

        # Hyperparameters that could be tuned 
        # (but are probably the best to use)
        self.activation_func = 'relu'
        self.optimizer = tf.train.AdamOptimizer

        # Logistics
        self.checkpoint_dir = checkpoint_dir
        self.filename = filename
        self.model_name = model_name
        self.output_type = output_type
        self.output_every_nth = 10

        # Extract the data from the filename
        self.data_loader = data_funcs.DataLoader(filename)
        self.input_size = self.data_loader.get_feature_size()
        if output_type == 'classification':
            self.output_size = self.data_loader.num_classes
        else:
            self.output_size = self.data_loader.num_outputs
        print "Input dimensions", self.input_size
        print "Number of classes/outputs", self.output_size

        # Set up tensorflow computation graph.
        self.graph = tf.Graph()
        self.build_graph()

        # Set up and initialize tensorflow session.
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)

        # Use for plotting evaluation.
        self.train_accuracies = []
        self.val_accuracies = []

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
                output_len = self.output_size
            else:
                output_len = self.layer_sizes[i]
                
            layer_weights = weight_variable([input_len, output_len],name='weights' + str(i))
            layer_biases = bias_variable([output_len], name='biases' + str(i))
            
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)
            sizes.append((str(input_len) + "x" + str(output_len), str(output_len)))
        
        print("Okay, making a neural net with the following structure:")
        print(sizes)

    def build_graph(self):
        """Constructs the tensorflow computation graph containing all variables
        that will be trained."""
        print '\nBuilding computation graph...'

        with self.graph.as_default():
            # Placeholders can be used to feed in different data during training time.
            self.tf_X = tf.placeholder(tf.float64, name="X") # features
            self.tf_Y = tf.placeholder(tf.int64, name="Y") # labels
            self.tf_dropout_prob = tf.placeholder(tf.float64) # Implements dropout

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
                            if self.activation_func == 'relu':
                                hidden = tf.nn.relu(hidden) 
                            else:
                                raise ValueError('That activation function has not been implemented.')

                            # Apply dropout
                            hidden = tf.nn.dropout(hidden, self.tf_dropout_prob) 
                return hidden
            self.run_network = run_network

            # Compute the loss function
            self.logits = run_network(self.tf_X)

            if self.output_type == 'classification':
                # Apply a softmax function to get probabilities, train this dist against targets with
                # cross entropy loss.
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, 
                                                                                          labels=self.tf_Y))

                 # Add weight decay regularization term to loss
                self.loss += self.weight_penalty * sum([tf.nn.l2_loss(w) for w in self.weights])

                # Code for making predictions and evaluating them.
                self.class_probabilities = tf.nn.softmax(self.logits)
                self.predictions = tf.argmax(self.class_probabilities, axis=1)
                self.correct_prediction = tf.equal(self.predictions, self.tf_Y)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            
            elif self.output_type == 'regression':
                # Apply mean squared error loss.
                self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.logits, self.tf_Y))))
                
                 # Add weight decay regularization term to loss
                self.loss = self.rmse + self.weight_penalty * sum([tf.nn.l2_loss(w) for w in self.weights])

            # Set up backpropagation computation!
            self.opt_step = self.optimizer(self.learning_rate).minimize(self.loss)

            # Necessary for tensorflow to build graph
            self.init = tf.global_variables_initializer()

    def train(self, num_steps=30000, output_every_nth=None):
        """Runs batches of training data through the model for a given
        number of steps.
        """
        if output_every_nth is not None:
            self.output_every_nth = output_every_nth

        with self.graph.as_default():
            # Used to save model checkpoints.
            self.saver = tf.train.Saver()

            for step in range(num_steps):
                X, Y = self.data_loader.get_train_batch(self.batch_size)
                feed_dict = {self.tf_X: X,
                             self.tf_Y: Y,
                             self.tf_dropout_prob: self.dropout_prob}
                if step % self.output_every_nth != 0:
                    # Train normally. Do not output results.
                    _ = self.session.run([self.opt_step], feed_dict)
                else:
                    # Train and save the training accuracy.
                    _, train_acc, = self.session.run([self.opt_step, self.accuracy], 
                                                     feed_dict)
                    
                    # Test on the validation set, save validation accuracy.
                    val_X, val_Y = self.data_loader.get_val_data()
                    feed_dict = {self.tf_X: val_X,
                                 self.tf_Y: val_Y,
                                 self.tf_dropout_prob: 1.0} # no dropout during evaluation
                    val_acc = self.session.run([self.accuracy], feed_dict)

                    print "Training iteration", step
                    print "\t Training accuracy", train_acc
                    print "\t Validation accuracy", val_acc
                    self.train_accuracies.append(train_acc)
                    self.val_accuracies.append(val_acc)

                    # Save a checkpoint of the model
                    self.saver.save(self.session, self.checkpoint_dir + self.model_name + '.ckpt', global_step=step)
    
    def predict(self, X, get_probabilities=False):
        feed_dict = {self.tf_X: X,
                     self.tf_dropout_prob: 1.0} # no dropout during evaluation
        probs, preds = self.session.run([self.class_probabilities, self.predictions], 
                                          feed_dict)
        if get_probabilities:
            return preds, probs
        else:
            return preds
    
    def plot_training_progress(self):
        x = [self.output_every_nth * i for i in np.arange(len(self.train_accuracies))]
        plt.figure()
        plt.plot(x,self.train_accuracies)
        plt.plot(x,self.val_accuracies)
        plt.legend(['Train', 'Validation'], loc='best')
        plt.xlabel('Training epoch')
        plt.ylabel('Accuracy')
        plt.show()

    def plot_binary_classification_data(self, with_decision_boundary=False):
        """ This function only works if there are two input features"""
        class1_X, class2_X = self.data_loader.get_train_binary_classification_data()
        
        plt.figure()
        plt.scatter(class1_X[:,0],class1_X[:,1], color='b')
        plt.scatter(class2_X[:,0],class2_X[:,1], color='r')
        
        if with_decision_boundary:
            # Make a mesh of points on which to make predictions
            mesh_step_size = .1
            x1_min = self.data_loader.train_X[:, 0].min() - 1
            x1_max = self.data_loader.train_X[:, 0].max() + 1
            x2_min = self.data_loader.train_X[:, 1].min() - 1
            x2_max = self.data_loader.train_X[:, 1].max() + 1
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, mesh_step_size),
                                   np.arange(x2_min, x2_max, mesh_step_size))
            
            # Make predictions for each point in the mesh
            Z = self.predict(np.c_[xx1.ravel(), xx2.ravel()])

            # Use matplotlib contour function to show decision boundary on mesh
            Z = Z.reshape(xx1.shape)
            plt.contour(xx1, xx2, Z, cmap=plt.cm.Paired)

        plt.show()

    def plot_regression_data(self):
        """sorted_val_x = sorted(dgp.val_X)
        mu, var = dgp.predict(sorted_val_x)

        plt.figure(figsize=(12, 6))
        plt.plot(dgp.df['X'], dgp.df['label'], 'x')
        plt.plot(sorted_val_x, mu, color='r', lw=2)
        plt.plot(sorted_val_x, mu + 2*np.sqrt(var), '--', color='r')
        plt.plot(sorted_val_x, mu - 2*np.sqrt(var), '--', color='r')"""
        print "THIS FUNCTION ISN'T FINISHED YET"


def weight_variable(shape,name):
	initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(shape[0])), dtype=tf.float64)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
	return tf.Variable(initial, name=name)

if __name__ == '__main__':
    print "THIS ISN'T IMPLEMENTED YET"

    parser = argparse.ArgumentParser(description='Open and query encrypted SQL files')
    parser.add_argument('-k', '--key', dest='key', required=True,
                        help='Private key to decrypt the files')
    args = parser.parse_args()
    
    t1 = time.time()
    conv_net = ArtistConvNet(invariance=invariance)
    conv_net.train_model()
    t2 = time.time()
    print "Finished training. Total time taken:", t2-t1
