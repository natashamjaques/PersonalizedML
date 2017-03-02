import pandas as pd

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