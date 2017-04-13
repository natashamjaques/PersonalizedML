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
import random

# Import data loading functions from parent directory.
CODE_PATH = os.path.dirname(os.getcwd())
sys.path.insert(1, CODE_PATH)
NUM_SUBJECTS = 42
PERSONALIZED_HIDDEN_LAYER_SIZE = 8

import data_funcs

def reload_files():
    reload(data_funcs)

class NeuralNetwork:
    def __init__(self, filename, model_name, layer_sizes=[128,64, 32], batch_size=25, 
                 learning_rate=.001, dropout_prob=0.9, weight_penalty=0.01, 
                 clip_gradients=True, model_type='regression', 
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
        #print dir(self.data_loader)
        self.input_size = self.data_loader.get_feature_size()
        if model_type == 'classification':
            print "\nPerforming classification."
            self.output_size = self.data_loader.num_classes
            self.metric_name = 'accuracy'
        else:
            print "\nPerforming regression."
            self.output_size = self.data_loader.num_outputs
            self.metric_name = 'RMSE'
        print "Input dimensions (number of features):", self.input_size
        print "Number of classes/outputs:", self.output_size
        
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

        #handle all the FC layers. For a 128x64x32 x (8 * 42) * (2 * 42) arch., ends at the 32 layer
        for i in range(len(self.layer_sizes)): 
            if i==0:
                input_len = self.input_size # X second dimension
            else:
                input_len = self.layer_sizes[i-1]


            output_len = self.layer_sizes[i]                
                
            layer_weights = weight_variable([input_len, output_len],name='weights' + str(i))
            layer_biases = bias_variable([output_len], name='biases' + str(i))
            
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)
            sizes.append((str(input_len) + "x" + str(output_len), str(output_len)))

        # set up hidden personalized layer weights
        input_len = self.layer_sizes[-1] # last hidden layer size        
        print('last non-personalized layer size was')
        print(input_len)

        
        self.p_weights0 = weight_variable([NUM_SUBJECTS, input_len, PERSONALIZED_HIDDEN_LAYER_SIZE],name='weights' + str(len(self.layer_sizes)))
        self.p_biases0 = bias_variable([NUM_SUBJECTS, PERSONALIZED_HIDDEN_LAYER_SIZE], name='biases' + str(len(self.layer_sizes)))


        # set up weights from personalized hidden layer to personalized output layer
        self.p_weights1 = weight_variable([NUM_SUBJECTS ,PERSONALIZED_HIDDEN_LAYER_SIZE, self.output_size],name='weights')
        self.p_biases1 = bias_variable([NUM_SUBJECTS, self.output_size], name='biases')

        print("Okay, making a neural net with the following structure:")
        print('Non-Personalized Layers:')
        print(sizes)
        print('Personalized Layers:')
        print(self.p_weights0.get_shape())
        print(' and ')
        print(self.p_weights1.get_shape())


    def build_graph(self):
        """Constructs the tensorflow computation graph containing all variables
        that will be trained."""
        print '\nBuilding computation graph...'

        with self.graph.as_default():
            # Placeholders can be used to feed in different data during training time.

            self.subject_num = tf.placeholder(tf.int32, name="subject_num") # subject number we are using in this minibatch
            self.tf_X = tf.placeholder(tf.float32, name="X") # features

            if self.model_type == 'classification':
                self.tf_Y = tf.placeholder(tf.int32, name="Y") # labels
            else: # regression
                self.tf_Y = tf.placeholder(tf.float32, name="Y") # labels
            self.tf_dropout_prob = tf.placeholder(tf.float32) # Implements dropout

            # Place the network weights/parameters that will be learned into the 
            # computation graph.
            self.initialize_network_weights()
            # Defines the actual network computations using the weights. 


            def run_network(input_X, subject_num):
                hidden0 = input_X        

                hidden1 = tf.matmul(hidden0, self.weights[0]) + self.biases[0]
                hidden1 = tf.nn.relu(hidden1)
                hidden1 = tf.nn.dropout(hidden1, self.tf_dropout_prob)

                print('hidden1 has shape')
                print(hidden1.get_shape())

                hidden2 = tf.matmul(hidden1, self.weights[1]) + self.biases[1]
                hidden2 = tf.nn.relu(hidden2)
                hidden2 = tf.nn.dropout(hidden2, self.tf_dropout_prob)
                print('hidden2 has shape')
                print(hidden2.get_shape())


                hidden3 = tf.matmul(hidden2, self.weights[2]) + self.biases[2]
                hidden3 = tf.nn.relu(hidden3)
                hidden3 = tf.nn.dropout(hidden3, self.tf_dropout_prob)
                print('hidden3 has shape')
                print(hidden3.get_shape())

                
                # will be a list of NUM_SUBJECTS elements, each of which should be a list of PERSONALIZED_HIDDEN_LAYER_SIZE values
                #hidden4_outputs = []

                #for i in range(0, NUM_SUBJECTS):
                hidden4 = tf.matmul(hidden3, self.p_weights0[subject_num,:,:]) + self.p_biases0[subject_num,:]
                hidden4 = tf.nn.relu(hidden4)
                hidden4 = tf.nn.dropout(hidden4, self.tf_dropout_prob)
                #hidden4_outputs.append(hidden4)

                # will be a list of NUM_SUBJECTS elements, each of which should be a list of PERSONALIZED_HIDDEN_LAYER_SIZE values                
                hidden5 = tf.matmul(hidden4, self.p_weights1[subject_num,:,:]) + self.p_biases1[subject_num,:]
                #hidden5_outputs.append(hidden5)

                print('returning output of hidden5!')
            

                return hidden5

            self.run_network = run_network

            # Compute the loss function
            print('TF_X shape')
            print(self.tf_X.get_shape())

            self.logits = run_network(self.tf_X, self.subject_num)

            #subject loss should be of type 1x2
            #self.subject_output = tf.gather(self.logits, self.subject_num)
            #print('subject output is of shape!')
            #print(self.subject_output.get_shape())

            
            if self.model_type == 'classification':
                # Apply a softmax function to get probabilities, train this dist against targets with
                # cross entropy loss.
                flat_labels = tf.reshape(self.tf_Y, [-1])
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, 
                                                                                          labels=flat_labels))

                 # Add weight decay regularization term to loss
                self.loss += self.weight_penalty * sum([tf.nn.l2_loss(w) for w in self.weights])

                # Code for making predictions and evaluating them.
                self.class_probabilities = tf.nn.softmax(self.logits)
                self.predictions = tf.argmax(self.class_probabilities, axis=1)
                self.correct_prediction = tf.equal(tf.cast(self.predictions, dtype=tf.int32), self.tf_Y)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            
            else: # regression
                # Apply mean squared error loss.
                self.squared_errors = tf.square(tf.subtract(self.logits, self.tf_Y)) #changed here to make personalized
                self.rmse = tf.sqrt(tf.reduce_mean(self.squared_errors))
                
                 # Add weight decay regularization term to loss
                self.loss = self.rmse + self.weight_penalty * sum([tf.nn.l2_loss(w) for w in self.weights])

                # Dont forget to regularize the personal weights!!
                self.loss = self.loss + self.weight_penalty * sum([tf.nn.l2_loss(self.p_weights0)])
                self.loss = self.loss + self.weight_penalty * sum([tf.nn.l2_loss(self.p_weights1)])

            # Set up backpropagation computation!
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            trainable_p_biases0 = self.p_biases0[self.subject_num]
            trainable_p_biases1 = self.p_biases1[self.subject_num]
            trainable_p_weights0 = self.p_weights0[self.subject_num]
            trainable_p_weights1 = self.p_weights1[self.subject_num]

            self.personal_training_params = [self.weights[0], self.weights[1], self.weights[2], self.biases[0], self.biases[1], self.biases[2],  
                self.p_weights0, self.p_weights1, self.p_biases0, self.p_biases1]
            

            self.params = tf.trainable_variables()
            #print('trainable variables are')
            #print(self.params)
            self.gradients = tf.gradients(self.loss, self.personal_training_params)
            if self.clip_gradients:
                self.gradients, _ = tf.clip_by_global_norm(self.gradients, 5)
            self.tf_optimizer = self.optimizer(self.learning_rate)
            self.opt_step = self.tf_optimizer.apply_gradients(zip(self.gradients, self.personal_training_params),
                                                              self.global_step)

            # Necessary for tensorflow to build graph
            self.init = tf.global_variables_initializer()

    def train(self, num_steps=30000, output_every_nth=None, subject_num=1):
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
                
                idx = random.randint(0, NUM_SUBJECTS-1)
                train_subject = np.arange(0, NUM_SUBJECTS)[idx] #we need a numpy int32 dtype


                X, Y = self.data_loader.get_personalized_train_batch(self.batch_size, train_subject+1) 
                #the data is 1 index'd in csv, but 0 indexd everywhere else
                
                feed_dict = {self.tf_X: X,
                             self.tf_Y: Y,
                             self.tf_dropout_prob: self.dropout_prob,
                             self.subject_num: train_subject}                
                
                # Update parameters in the direction of the gradient computed by
                # the optimizer.
                
                _ = self.session.run([self.opt_step], feed_dict)

                # Output/save the training and validation performance every few steps.
                if step % self.output_every_nth == 0:
                    # Grab a batch of validation data too.
                    val_X, val_Y = self.data_loader.get_personalized_val_data(train_subject)
                    val_feed_dict = {self.tf_X: val_X,
                                     self.tf_Y: val_Y,
                                     self.tf_dropout_prob: 1.0, # no dropout during evaluation
                                     self.subject_num: train_subject} 

                    if self.model_type == 'classification':
                        train_score, loss = self.session.run([self.accuracy, self.loss], feed_dict)
                        val_score, loss = self.session.run([self.accuracy, self.loss], val_feed_dict)
                    else: # regression
                        train_score, loss = self.session.run([self.rmse, self.loss], feed_dict)
                        val_score, loss = self.session.run([self.rmse, self.loss],  val_feed_dict)
                    
                    print "Training iteration", step
                    print "\t Training", self.metric_name, train_score
                    print "\t Validation", self.metric_name, val_score
                    print "\t Loss", loss
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
        scores = []
        for i in range(0, NUM_SUBJECTS):
            X,Y = self.data_loader.get_personalized_val_data(i+1)
            val_score = self.get_performance_on_data(X,Y,i)
            print val_score
            scores.append(val_score)
        print "Final", self.metric_name, "on validation data is:", np.mean(scores)
        print scores
        return np.mean(scores)
        
    def test_on_test(self):
        """Returns performance on the model's test set."""
        print "WARNING! Only test on the test set when you have finished choosing all of your hyperparameters!"
        print "\tNever use the test set to choose hyperparameters!!!"
        score = self.get_performance_on_data(self.data_loader.test_X,
                                             self.data_loader.test_Y)
        print "Final", self.metric_name, "on test data is:", score
        return score

    def get_performance_on_data(self, X, Y, i):
        """Returns the model's performance on input data X and targets Y."""
        feed_dict = {self.tf_X: X,
                     self.tf_Y: Y,
                     self.tf_dropout_prob: 1.0, # no dropout during evaluation
                     self.subject_num: i}
        
        if self.model_type == 'classification':
            score = self.session.run(self.accuracy, feed_dict)
        else: # regression
            score = self.session.run(self.rmse, feed_dict)
        
        return score

def weight_variable(shape,name):
    """Initializes a tensorflow weight variable with random
    values centered around 0.
    """
    initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(shape[0])), dtype=tf.float32)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    """Initializes a tensorflow bias variable to a small constant value."""
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
