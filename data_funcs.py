""" This file provides functions for loading data from a saved 
    .csv file. 

    The code assumes that the file contains at least one column 
    with 'label' in the name, containing an outcome we are trying
    to classify. E.g. if we are trying to predict if someone is 
    happy or not, we might have a column 'happy_label' that has 
    values that are either 0 or 1. 
    
    We also assume the file contains a column named 'dataset' 
    containing the dataset the example belongs to, which can be 
    either 'Train', 'Val', or 'Test'. Remember, it is important 
    to never train or choose parameters based on the test set!
"""

import pandas as pd
import numpy as np

USE_PERSONAL_FEATS = True

class DataLoader:
    def __init__(self, filename):
        self.load_and_process_data(filename)

    def load_and_process_data(self, file_path, suppress_output=False):
        df = pd.DataFrame.from_csv(file_path)

        if USE_PERSONAL_FEATS:
            self.wanted_feats = [x for x in df.columns.values if 'label' not in x and 'dataset' not in x]            
        else:
            self.wanted_feats = [x for x in df.columns.values if 'label' not in x and 'dataset' not in x and 'personal' not in x and 'subject_num' not in x and 'test_time' not in x]

        self.wanted_labels = [y for y in df.columns.values if 'label' in y]

        self.df = normalize_fill_df(df, self.wanted_feats, self.wanted_labels, 
                                    suppress_output=suppress_output, remove_cols=True)
        
        self.num_outputs = len(self.wanted_labels)
        if len(self.wanted_labels) == 1:
            self.num_classes = len(self.df[self.wanted_labels[0]].unique())

        self.train_X, self.train_Y = get_matrices_for_dataset(self.df, self.wanted_feats, 
                                                            self.wanted_labels, 'Train')
        if not suppress_output: print len(self.train_X), "rows in training data"
        
        self.val_X, self.val_Y = get_matrices_for_dataset(self.df, self.wanted_feats, 
                                                        self.wanted_labels, 'Val')
        #print('val!')
        #print(self.val_X, self.val_Y)
        if not suppress_output: print len(self.val_X), "rows in validation data"
        
        self.test_X, self.test_Y = get_matrices_for_dataset(self.df, self.wanted_feats, 
                                                            self.wanted_labels, 'Test')
        if not suppress_output: print len(self.test_X), "rows in testing data"
    
    def get_personalized_train_batch(self, batch_size, subject_num):

        personal_train_X, personal_train_Y = get_personalized_matrices_for_dataset(self.df, self.wanted_feats, 
                                                            self.wanted_labels, 'Train', subject_num)
        
        idx = np.random.choice(len(personal_train_X), size=batch_size)
        return personal_train_X[idx], personal_train_Y[idx]

    def get_train_batch(self, batch_size):
        idx = np.random.choice(len(self.train_X), size=batch_size)
        return self.train_X[idx], self.train_Y[idx]

    def get_personalized_val_data(self, subject_num):
        personal_val_X, personal_val_Y = get_personalized_matrices_for_dataset(self.df, self.wanted_feats, 
                                                            self.wanted_labels, 'Val', subject_num)
        return personal_val_X, personal_val_Y

    def get_val_data(self):
        return self.val_X, self.val_Y

    def get_feature_size(self):
        return np.shape(self.train_X)[1]
    
    def get_train_binary_classification_data(self):
        train_df = self.df[self.df['dataset']=='Train']
        class1_df = train_df[train_df['label']==0]
        class2_df = train_df[train_df['label']==1]

        class1_X = class1_df[self.wanted_feats].astype(float).as_matrix()
        class1_X = convert_matrix_tf_format(class1_X)

        class2_X = class2_df[self.wanted_feats].astype(float).as_matrix()
        class2_X = convert_matrix_tf_format(class2_X)

        return class1_X, class2_X


def normalize_fill_df(data_df, wanted_feats, wanted_labels, suppress_output=False, remove_cols=True):
    data_df = normalize_columns(data_df, wanted_feats)
    if remove_cols:
        data_df, wanted_feats = remove_null_cols(data_df, wanted_feats)

    if not suppress_output: print "Original data length was", len(data_df)
    data_df = data_df.dropna(subset=wanted_labels, how='any')
    if not suppress_output: print "After dropping rows with nan in any label column, length is", len(data_df)

    data_df = data_df.fillna(0) #if dataset is already filled, won't do anything

    # Shuffle data
    data_df = data_df.sample(frac=1)

    return data_df

def get_personalized_matrices_for_dataset(data_df, wanted_feats, wanted_labels, dataset, subject_num, single_output=False):
    set_df = data_df[(data_df['dataset']==dataset) & (data_df['subject_num']==subject_num)]
    
    X = set_df[wanted_feats].astype(float).as_matrix()
    
    if single_output:
        y = set_df[wanted_labels[0]].tolist()
    else:
        y = set_df[wanted_labels].as_matrix()
    
    X = convert_matrix_tf_format(X)
    y = np.atleast_2d(np.asarray(y))

    return X,y

def get_matrices_for_dataset(data_df, wanted_feats, wanted_labels, dataset, single_output=False):
    set_df = data_df[data_df['dataset']==dataset]
    
    X = set_df[wanted_feats].astype(float).as_matrix()
    
    if single_output:
        y = set_df[wanted_labels[0]].tolist()
    else:
        y = set_df[wanted_labels].as_matrix()
    
    X = convert_matrix_tf_format(X)
    y = np.atleast_2d(np.asarray(y))

    return X,y

def convert_matrix_tf_format(X):
    X = np.asarray(X)
    X = X.astype(np.float64)
    return X

def normalize_columns(df, wanted_feats):
    train_df = df[df['dataset']=='Train']
    for feat in wanted_feats:

        if not feat == 'subject_num':
            train_mean = np.mean(train_df[feat].dropna().tolist())
            train_std = np.std(train_df[feat].dropna().tolist())
            zscore = lambda x: (x - train_mean) / train_std
            df[feat] = df[feat].apply(zscore)
        else:
            print 'skipping normalization for subject_num'
    return df

def find_null_columns(df, features):
    df_len = len(df)
    bad_feats = []
    for feat in features:
        null_len = len(df[df[feat].isnull()])
        if df_len == null_len:
            bad_feats.append(feat)
    return bad_feats

def remove_null_cols(df, features):
    '''Must check if a column is completely null in any of the datasets. Then it will remove it'''
    train_df = df[df['dataset']=='Train']
    test_df = df[df['dataset']=='Test']
    val_df = df[df['dataset']=='Val']

    null_cols = find_null_columns(train_df,features)
    null_cols_test= find_null_columns(test_df,features)
    null_cols_val = find_null_columns(val_df,features)

    if len(null_cols) > 0 or len(null_cols_test) > 0 or len(null_cols_val) > 0:
        for feat in null_cols_test:
            if feat not in null_cols:
                null_cols.append(feat)
        for feat in null_cols_val:
            if feat not in null_cols:
                null_cols.append(feat)
        print "Found", len(null_cols), "columns that were completely null. Removing", null_cols

        df = dropCols(df,null_cols)
        for col in null_cols:
            features.remove(col)
    return df, features