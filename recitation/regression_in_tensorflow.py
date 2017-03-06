""" Code that implements a basic regression function in tensorflow.

    To perform classification, this code can be used as a Logistic 
    Regression model (with a sigmoid loss function). To perform 
    regression, this code can be used as a Linear Regression model 
    with a least squares loss. 

    What this code implements is equivalent to a single neuron in 
    a neural network.

    The code uses Stochastic Gradient Descent (SGD) to train the
    model. 
    
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

def reload_files():
    reload(data_funcs)

class TFRegressor:
    def __init__(self, filename, batch_size=20, learning_rate=.001, weight_penalty=0.0, 
                 model_type='classification'):
        '''Initialize the class by loading the required datasets and building the 
        tensorlfow computation graph.

        Args:
            filename: a file containing the data.
            batch_size: number of training examples in each training batch. 
            learning_rate: the initial learning rate used in stochastic 
                gradient descent.
            weight_penalty: the coefficient of the L2 weight regularization
                applied to the loss function. Set to > 0.0 to apply weight 
                regularization, 0.0 to remove.
            model_type: the type of regression. Either 'classification' in 
                which case the model is a Logistic Regression classifier, or 
                'regression', in which it is a linear regression model.
        '''
        # Save the hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_penalty = weight_penalty 

        # Optimization function to use to train the model. 
        # It's widely accepted that the Adam optimizer rules ---
        # I mean, is state-of-the-art --- but you could choose 
        # others, like tf.train.AdagradOptimizer, or even
        # tf.train.GradientDescentOptimizer.
        self.optimizer = tf.train.AdamOptimizer

        # Logistics
        self.model_type = model_type
        self.output_every_nth = 10 # save performance every n steps

        # Extract the data from the filename
        self.data_loader = data_funcs.DataLoader(filename)
        self.input_size = self.data_loader.get_feature_size()
        if model_type == 'classification':
            print "\nPerforming classification."
            self.output_size = 1 # limited to binary classification
            self.metric_name = 'accuracy'
        else:
            print "\nPerforming regression."
            self.output_size = self.data_loader.num_outputs
            self.metric_name = 'RMSE'
        
        # Set up tensorflow computation graph.
        self.graph = tf.Graph()
        self.build_graph()

        # Set up and initialize tensorflow session.
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)

        # Use for plotting evaluation.
        self.train_metrics = []
        self.val_metrics = []

    def build_graph(self):
        """Constructs the tensorflow computation graph containing all variables
        that will be trained."""
        print '\nBuilding computation graph...'

        with self.graph.as_default():
            # Placeholders can be used to feed in different data during training time.
            self.tf_X = tf.placeholder(tf.float64, name="X") # features
            self.tf_Y = tf.placeholder(tf.float64, name="Y") # labels

            # The weights/parameters that will be learned.
            self.W = weight_variable([self.input_size, self.output_size],name='weights')
            self.b = bias_variable([self.output_size], name='biases')
            
            self.logits = tf.matmul(self.tf_X, self.W) + self.b
            self.flat_logits = tf.reshape(self.logits, [-1])

            if self.model_type == 'classification':
                # Train the model using sigmoid loss for logistic regression.
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.flat_logits, 
                                                                                   labels=self.tf_Y))

                 # Add weight decay regularization term to loss
                # TODO: ADD REGULARIZATION HERE!!

                # Code for making predictions and evaluating them.
                self.probabilities = tf.nn.sigmoid(self.flat_logits)
                self.predictions = tf.round(self.probabilities)
                self.correct_prediction = tf.equal(self.predictions, self.tf_Y)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            
            else: # regression
                # Apply mean squared error loss.
                self.squared_errors = tf.square(tf.subtract(self.flat_logits, self.tf_Y))
                self.rmse = tf.sqrt(tf.reduce_mean(self.squared_errors))
                
                # Add weight decay regularization term to loss
                # TODO: ADD REGULARIZATION HERE!!
                self.loss = self.rmse

            # Set up backpropagation computation!
            self.opt_step = self.optimizer(self.learning_rate).minimize(self.loss)

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
            for step in range(num_steps):
                # Grab a batch of data to feed into the placeholders in the graph.
                X, Y = self.data_loader.get_train_batch(self.batch_size)
                feed_dict = {self.tf_X: X, self.tf_Y: Y}
                
                # Update parameters in the direction of the gradient computed by
                # the optimizer.
                _ = self.session.run([self.opt_step], feed_dict)

                # Output/save the training and validation performance every few steps.
                if step % self.output_every_nth == 0:
                    # Grab a batch of validation data too.
                    val_X, val_Y = self.data_loader.get_val_data()
                    val_feed_dict = {self.tf_X: val_X, self.tf_Y: val_Y}

                    if self.model_type == 'classification':
                        train_score = self.session.run(self.accuracy, feed_dict)
                        val_score = self.session.run(self.accuracy, val_feed_dict)
                    else: # regression
                        train_score = self.session.run(self.rmse, feed_dict)
                        val_score = self.session.run(self.rmse, val_feed_dict)
                    
                    print "Training iteration", step
                    print "\t Training", self.metric_name, train_score
                    print "\t Validation", self.metric_name, val_score
                    self.train_metrics.append(train_score)
                    self.val_metrics.append(val_score)

    def predict(self, X, get_probabilities=False):
        """Gets the network's predictions for some new data X
        
        Args: 
            X: a matrix of data in the same format as the training
                data. 
            get_probabilities: a boolean that if true, will cause 
                the function to return the model's computed 
                probabilities in addition to its predictions. Only 
                works for classification.
        Returns:
            integer class predictions if the model is doing 
            classification, otherwise float predictions if the 
            model is doing regression.
        """
        feed_dict = {self.tf_X: X}
        
        if self.model_type == 'classification':
            probs, preds = self.session.run([self.probabilities, self.predictions], 
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
        print "Final", self.metric_name, "on validation data is:", score
        return score
        
    def test_on_test(self):
        """Returns performance on the model's test set."""
        print "WARNING! Only test on the test set when you have finished choosing all of your hyperparameters!"
        print "\tNever use the test set to choose hyperparameters!!!"
        score = self.get_performance_on_data(self.data_loader.test_X,
                                             self.data_loader.test_Y)
        print "Final", self.metric_name, "on test data is:", score
        return score

    def get_performance_on_data(self, X, Y):
        """Returns the model's performance on input data X and targets Y."""
        feed_dict = {self.tf_X: X,self.tf_Y: Y}
        
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
