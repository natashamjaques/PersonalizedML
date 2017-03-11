""" Code that implements a basic fully-connected neural network 
    in Tensorflow. 

    Neural networks have many parameters that can be tuned. You can 
    tweak the parameters when running this code by feeding them 
    into the class constructor.

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
import os
import math
import time

# Import data loading functions from parent directory.
CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)
import data_funcs
import imp

def reload_files():
    imp.reload(data_funcs)

class NeuralNetwork:
    def __init__(self, filename, model_name, layer_sizes=[128,64], batch_size=10, 
                 learning_rate=.01, dropout_prob=1.0, weight_penalty=0.0, 
                 clip_gradients=True, model_type='classification', 
                 checkpoint_dir='./saved_models/'):
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
            clip_gradients: a bool indicating whether or not to clip gradients. 
                This is effective in preventing very large gradients from skewing 
                training, and preventing your loss from going to inf or nan. 
            model_type: the type of output prediction. Either 'classification'
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
        self.clip_gradients = clip_gradients
        self.activation_func = 'relu'
        self.optimizer = tf.train.AdamOptimizer

        # Logistics
        self.checkpoint_dir = checkpoint_dir
        self.filename = filename
        self.model_name = model_name
        self.model_type = model_type
        self.output_every_nth = 10

        # Extract the data from the filename
        self.data_loader = data_funcs.DataLoader(filename)
        self.input_size = self.data_loader.get_feature_size()
        if model_type == 'classification':
            print("\nPerforming classification.")
            self.output_size = self.data_loader.num_classes
            self.metric_name = 'accuracy'
        else:
            print("\nPerforming regression.")
            self.output_size = self.data_loader.num_outputs
            self.metric_name = 'RMSE'
        print("Input dimensions (number of features):", self.input_size)
        print("Number of classes/outputs:", self.output_size)
        
        # Set up tensorflow computation graph.
        self.graph = tf.Graph()
        self.build_graph()

        # Set up and initialize tensorflow session.
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)

        # Use for plotting evaluation.
        self.train_metrics = []
        self.val_metrics = []

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
        print('\nBuilding computation graph...')

        with self.graph.as_default():
            # Placeholders can be used to feed in different data during training time.
            self.tf_X = tf.placeholder(tf.float64, name="X") # features
            if self.model_type == 'classification':
                self.tf_Y = tf.placeholder(tf.int64, name="Y") # labels
            else: # regression
                self.tf_Y = tf.placeholder(tf.float64, name="Y") # labels
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
                            # Could add more activation functions like sigmoid here
                            # If no activation is specified, none will be applied

                            # Apply dropout
                            hidden = tf.nn.dropout(hidden, self.tf_dropout_prob) 
                return hidden
            self.run_network = run_network

            # Compute the loss function
            self.logits = run_network(self.tf_X)

            if self.model_type == 'classification':
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
            
            else: # regression
                # Apply mean squared error loss.
                self.squared_errors = tf.square(tf.subtract(tf.reshape(self.logits, [-1]), self.tf_Y))
                self.rmse = tf.sqrt(tf.reduce_mean(self.squared_errors))
                
                 # Add weight decay regularization term to loss
                self.loss = self.rmse + self.weight_penalty * sum([tf.nn.l2_loss(w) for w in self.weights])

            # Set up backpropagation computation!
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.loss, self.params)
            if self.clip_gradients:
                self.gradients, _ = tf.clip_by_global_norm(self.gradients, 5)
            self.tf_optimizer = self.optimizer(self.learning_rate)
            self.opt_step = self.tf_optimizer.apply_gradients(zip(self.gradients, self.params),
                                                              self.global_step)

            # Necessary for tensorflow to build graph
            self.init = tf.global_variables_initializer()

    def train(self, num_steps=30000, output_every_nth=None):
        """Trains using stochastic gradient descent (SGD). 
        
        Runs batches of training data through the model for a given
        number of steps.

        Note that if you set the class's batch size to the number
        of points in the training data, you would be doing gradient
        descent rather than SGD. SGD is preferred since it has a 
        strong regularizing effect.
        """
        if output_every_nth is not None:
            self.output_every_nth = output_every_nth

        with self.graph.as_default():
            # Used to save model checkpoints.
            self.saver = tf.train.Saver()

            for step in range(num_steps):
                # Grab a batch of data to feed into the placeholders in the graph.
                X, Y = self.data_loader.get_train_batch(self.batch_size)
                feed_dict = {self.tf_X: X,
                             self.tf_Y: Y,
                             self.tf_dropout_prob: self.dropout_prob}
                
                # Update parameters in the direction of the gradient computed by
                # the optimizer.
                _ = self.session.run([self.opt_step], feed_dict)

                # Output/save the training and validation performance every few steps.
                if step % self.output_every_nth == 0:
                    # Grab a batch of validation data too.
                    val_X, val_Y = self.data_loader.get_val_data()
                    val_feed_dict = {self.tf_X: val_X,
                                     self.tf_Y: val_Y,
                                     self.tf_dropout_prob: 1.0} # no dropout during evaluation

                    if self.model_type == 'classification':
                        train_score = self.session.run(self.accuracy, feed_dict)
                        val_score = self.session.run(self.accuracy, val_feed_dict)
                    else: # regression
                        train_score = self.session.run(self.rmse, feed_dict)
                        val_score = self.session.run(self.rmse, val_feed_dict)
                    
                    print("Training iteration", step)
                    print("\t Training", self.metric_name, train_score)
                    print("\t Validation", self.metric_name, val_score)
                    self.train_metrics.append(train_score)
                    self.val_metrics.append(val_score)

                    # Save a checkpoint of the model
                    self.saver.save(self.session, self.checkpoint_dir + self.model_name + '.ckpt', global_step=step)
    
    def predict(self, X, get_probabilities=False):
        """Gets the network's predictions for some new data X
        
        Args: 
            X: a matrix of data in the same format as the training
                data. 
            get_probabilities: a boolean that if true, will cause 
                the function to return the model's computed softmax
                probabilities in addition to its predictions. Only 
                works for classification.
        Returns:
            integer class predictions if the model is doing 
            classification, otherwise float predictions if the 
            model is doing regression.
        """
        feed_dict = {self.tf_X: X,
                     self.tf_dropout_prob: 1.0} # no dropout during evaluation
        
        if self.model_type == 'classification':
            probs, preds = self.session.run([self.class_probabilities, self.predictions], 
                                            feed_dict)
            if get_probabilities:
                return preds, probs
            else:
                return preds
        else: # regression
            return self.session.run(self.logits, feed_dict)
    
    def plot_training_progress(self):
        """Plots the training and validation performance as evaluated 
        throughout training."""
        x = [self.output_every_nth * i for i in np.arange(len(self.train_metrics))]
        plt.figure()
        plt.plot(x,self.train_metrics)
        plt.plot(x,self.val_metrics)
        plt.legend(['Train', 'Validation'], loc='best')
        plt.xlabel('Training epoch')
        plt.ylabel(self.metric_name)
        plt.show()

    def plot_binary_classification_data(self, with_decision_boundary=False):
        """Plots the data from each of two binary classes with two different
        colours. If with_decision_boundary is set to true, also plots the 
        decision boundary learned by the model.
        
        Note: This function only works if there are two input features.
        """
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

    def plot_regression_data(self, with_decision_boundary=False):
        """Plots input regression data. If with_decision_boundary is set 
        to true, also plots the regression function learned by the model.
        
        Note: This function only works if there is one input feature.
        """
        plt.figure()
        plt.scatter(self.data_loader.train_X, self.data_loader.train_Y)
        
        if with_decision_boundary:
            sorted_x = sorted(self.data_loader.train_X)
            preds = self.predict(sorted_x)
            plt.plot(sorted_x, preds, color='r', lw=2)
        
        plt.show()

    def test_on_validation(self):
        """Returns performance on the model's validation set."""
        score = self.get_performance_on_data(self.data_loader.val_X,
                                             self.data_loader.val_Y)
        print("Final", self.metric_name, "on validation data is:", score)
        return score
        
    def test_on_test(self):
        """Returns performance on the model's test set."""
        print("WARNING! Only test on the test set when you have finished choosing all of your hyperparameters!")
        print("\tNever use the test set to choose hyperparameters!!!")
        score = self.get_performance_on_data(self.data_loader.test_X,
                                             self.data_loader.test_Y)
        print "Final", self.metric_name, "on test data is:", score
        return score

    def get_performance_on_data(self, X, Y):
        """Returns the model's performance on input data X and targets Y."""
        feed_dict = {self.tf_X: X,
                     self.tf_Y: Y,
                     self.tf_dropout_prob: 1.0} # no dropout during evaluation
        
        if self.model_type == 'classification':
            score = self.session.run(self.accuracy, feed_dict)
        else: # regression
            score = self.session.run(self.rmse, feed_dict)
        
        return score

def weight_variable(shape,name):
    """Initializes a tensorflow weight variable with random
    values centered around 0.
    """
    initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(shape[0])), dtype=tf.float64)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    """Initializes a tensorflow bias variable to a small constant value."""
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, name=name)
