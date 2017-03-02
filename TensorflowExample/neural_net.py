import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import sys
import math
import time

DATA_PATH = 'art_data/'
DATA_FILE = DATA_PATH + 'art_data.pickle'
IMAGE_SIZE = 50
NUM_CHANNELS = 3
NUM_LABELS = 11
INCLUDE_TEST_SET = False

class NeuralNetwork:
    def __init__(self, filename):
        '''Initialize the class by loading the required datasets 
        and building the graph.

        Args:
            filename: a file containing the data.'''

        self.input_size

        # Hyperparameters that should be tuned
        self.batch_size = 10
        self.learning_rate = 0.01
        self.layer_sizes = [128,64]
        self.dropout_prob = 1.0 # set to < 1.0 to apply dropout, 1.0 to remove
        self.weight_penalty = 0.0 # set to > 0.0 to apply weight penalty, 0.0 to remove

        # Set up tensorflow computation graph.
        self.graph = tf.Graph()
        self.build_graph()

    def initialize_network_weights(self):
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
        print '\nBuilding computation graph...'

        with self.graph.as_default():
            # Input data
            tf_train_batch = tf.placeholder(
                tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_LABELS))
            tf_valid_dataset = tf.constant(self.val_X)
            tf_test_dataset = tf.placeholder(
                tf.float32, shape=[len(self.val_X), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
            tf_train_dataset = tf.placeholder(
                tf.float32, shape=[len(self.train_X), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

            # Implement dropout
            dropout_keep_prob = tf.placeholder(tf.float32)

            # Network weights/parameters that will be learned
            layer1_weights = tf.Variable(tf.truncated_normal(
                [layer1_filter_size, layer1_filter_size, NUM_CHANNELS, layer1_depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([layer1_depth]))
            layer1_feat_map_size = int(math.ceil(float(IMAGE_SIZE) / layer1_stride))
            if pooling:
                layer1_feat_map_size = int(math.ceil(float(layer1_feat_map_size) / layer1_pool_stride))

            layer2_weights = tf.Variable(tf.truncated_normal(
                [layer2_filter_size, layer2_filter_size, layer1_depth, layer2_depth], stddev=0.1))
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[layer2_depth]))
            layer2_feat_map_size = int(math.ceil(float(layer1_feat_map_size) / layer2_stride))
            if pooling:
                layer2_feat_map_size = int(math.ceil(float(layer2_feat_map_size) / layer2_pool_stride))

            layer3_weights = tf.Variable(tf.truncated_normal(
                [layer2_feat_map_size * layer2_feat_map_size * layer2_depth, layer3_num_hidden], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[layer3_num_hidden]))

            new_layer_weights = tf.Variable(tf.truncated_normal(
                [layer3_num_hidden, new_layer_num_hidden], stddev=0.1))
            new_layer_biases = tf.Variable(tf.constant(1.0, shape=[new_layer_num_hidden]))

            layer4_weights = tf.Variable(tf.truncated_normal(
              [new_layer_num_hidden, NUM_LABELS], stddev=0.1))
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

            # Model
            def network_model(data):
                '''Define the actual network architecture'''

                # Layer 1
                conv1 = tf.nn.conv2d(data, layer1_weights, [1, layer1_stride, layer1_stride, 1], padding='SAME')
                hidden = tf.nn.relu(conv1 + layer1_biases)

                if pooling:
                    hidden = tf.nn.max_pool(hidden, ksize=[1, layer1_pool_filter_size, layer1_pool_filter_size, 1], 
                                       strides=[1, layer1_pool_stride, layer1_pool_stride, 1],
                                        padding='SAME', name='pool1')
                
                # Layer 2
                conv2 = tf.nn.conv2d(hidden, layer2_weights, [1, layer2_stride, layer2_stride, 1], padding='SAME')
                hidden = tf.nn.relu(conv2 + layer2_biases)

                if pooling:
                    hidden = tf.nn.max_pool(hidden, ksize=[1, layer2_pool_filter_size, layer2_pool_filter_size, 1], 
                                       strides=[1, layer2_pool_stride, layer2_pool_stride, 1],
                                        padding='SAME', name='pool2')
                
                # Layer 3
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
                hidden = tf.nn.dropout(hidden, dropout_keep_prob)

                # new layer
                hidden = tf.nn.relu(tf.matmul(hidden, new_layer_weights) + new_layer_biases)
                hidden = tf.nn.dropout(hidden, dropout_keep_prob)

                # Layer 4 
                output = tf.matmul(hidden, layer4_weights) + layer4_biases
                return output

            # Training computation
            logits = network_model(tf_train_batch)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

            # Add weight decay penalty
            loss = loss + weight_decay_penalty([layer3_weights, layer4_weights], weight_penalty)

            # Optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            # Predictions for the training, validation, and test data.
            batch_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(network_model(tf_valid_dataset))
            test_prediction = tf.nn.softmax(network_model(tf_test_dataset))
            train_prediction = tf.nn.softmax(network_model(tf_train_dataset))

            def train_model(num_steps=num_training_steps):
                '''Train the model with minibatches in a tensorflow session'''
                with tf.Session(graph=self.graph) as session:
                    tf.initialize_all_variables().run()
                    print 'Initializing variables...'
                    
                    for step in range(num_steps):
                        offset = (step * batch_size) % (self.train_Y.shape[0] - batch_size)
                        batch_data = self.train_X[offset:(offset + batch_size), :, :, :]
                        batch_labels = self.train_Y[offset:(offset + batch_size), :]
                        
                        # Data to feed into the placeholder variables in the tensorflow graph
                        feed_dict = {tf_train_batch : batch_data, tf_train_labels : batch_labels, 
                                     dropout_keep_prob: dropout_prob}
                        _, l, predictions = session.run(
                          [optimizer, loss, batch_prediction], feed_dict=feed_dict)
                        if (step % 100 == 0):
                            train_preds = session.run(train_prediction, feed_dict={tf_train_dataset: self.train_X,
                                                                           dropout_keep_prob : 1.0})
                            val_preds = session.run(valid_prediction, feed_dict={dropout_keep_prob : 1.0})
                            print ''
                            print('Batch loss at step %d: %f' % (step, l))
                            print('Batch training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                            print('Validation accuracy: %.1f%%' % accuracy(val_preds, self.val_Y))
                            print('Full train accuracy: %.1f%%' % accuracy(train_preds, self.train_Y))

                    # This code is for the final question
                    if self.invariance:
                        print "\n Obtaining final results on invariance sets!"
                        sets = [self.val_X, self.translated_val_X, self.bright_val_X, self.dark_val_X, 
                                self.high_contrast_val_X, self.low_contrast_val_X, self.flipped_val_X, 
                                self.inverted_val_X,]
                        set_names = ['normal validation', 'translated', 'brightened', 'darkened', 
                                     'high contrast', 'low contrast', 'flipped', 'inverted']
                        
                        for i in range(len(sets)):
                            preds = session.run(test_prediction, 
                                feed_dict={tf_test_dataset: sets[i], dropout_keep_prob : 1.0})
                            print 'Accuracy on', set_names[i], 'data: %.1f%%' % accuracy(preds, self.val_Y)

                            # save final preds to make confusion matrix
                            if i == 0:
                                self.final_val_preds = preds 
            
            # save train model function so it can be called later
            self.train_model = train_model

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

    def load_invariance_datasets(self):
        with open(DATA_PATH + 'invariance_art_data.pickle', 'rb') as f:
            save = pickle.load(f)
            self.translated_val_X = save['translated_val_data']
            self.flipped_val_X = save['flipped_val_data']
            self.inverted_val_X = save['inverted_val_data']
            self.bright_val_X = save['bright_val_data']
            self.dark_val_X = save['dark_val_data']
            self.high_contrast_val_X = save['high_contrast_val_data']
            self.low_contrast_val_X = save['low_contrast_val_data']
            del save  

def weight_decay_penalty(weights, penalty):
    return penalty * sum([tf.nn.l2_loss(w) for w in weights])

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

if __name__ == '__main__':
    invariance = False
    if len(sys.argv) > 1 and sys.argv[1] == 'invariance':
        print "Testing finished model on invariance datasets!"
        invariance = True
    
    t1 = time.time()
    conv_net = ArtistConvNet(invariance=invariance)
    conv_net.train_model()
    t2 = time.time()
    print "Finished training. Total time taken:", t2-t1
