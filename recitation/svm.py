""" Code that provides a wrapper around sklearn SVC class to make it easier to use.

    This implements a soft-margin SVM, which is able to mis-classify some points
    while incurring a penalty for those points in the objective function. The 
    parameter C controls how heavily the classifier is penalized for mis-classified
    points.
    
    SVMs depend on the kernel function, which maps the original feature space into 
    a new feature space. In this implementation, the kernel can be:
        - linear: Just the dot product
        - rbf: Gaussian radial basis function: exp(-gamma * ||x - x'||2) 
               The parameter to tune is gamma. A large gamma means a more complex
               function (high variance), while a small gamma means a smoother
               decision boundary (high bias).
        - poly: A polynomial kernel: raises the dot product kernel to the power of \
               poly_degree.

    For more information on support vector machines in scikit learn,
    see: http://scikit-learn.org/stable/modules/svm.html

    For more information about the sklearn SVC class, see:
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt 

# Import data loading functions from parent directory.
CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)
import data_funcs

def reload_files():
    reload(data_funcs)

class SVM:
    def __init__(self, filename, C=1.0, kernel='linear', gamma=.01, poly_degree=3, 
                 max_iter=-1, tolerance=0.001):
        """Initialize the class by loading the required data and setting the parameters

        Args:
            filename: a file containing the data.
            C: a float for the soft-margin SVM misclassification penalty.
            kernel: the type of kernel to use. Can be 'linear', 'rbf', or 'poly'. 
            gamma: a float kernel parameter. 
            poly_degree: the degree of the polynomial used in the 'poly' kernel. 
            max_iter: the maximum number of iterations to run when training.
            tolerance: a float epsilon value. If the loss function changes by only
                tolerance or less, the funtion will stop training.
        """

        # Load the data.
        self.data_loader = data_funcs.DataLoader(filename)

        # Set the parameters.
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.poly_degree = poly_degree
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.classifier = None

    def train(self):
        """Initializes a classifier with right parameters and fits it according to its optimization
        objective."""
        self.classifier = SVC(C=self.C, kernel=self.kernel, probability=True, gamma=self.gamma, 
                              degree=self.poly_degree, max_iter=self.max_iter, tol=self.tolerance)
        self.classifier.fit(self.data_loader.train_X, self.data_loader.train_Y)

    def predict(self, X):
        """Gets the classifier's predicted class for each data point in every row of X."""
        return self.classifier.predict(X)

    def get_accuracy(self, X, Y):
        """Returns the classification accuracy of the classifier's predictions on data points in X 
        compared to their ground truth labels in Y. """
        return self.classifier.score(X, Y)

    def get_fpr_and_tpr(self,X,Y):
        """Gets the false positive rate and true positive rate from the classifier's predictions on
        data in X as compared to the ground truth labels in Y."""
        probas_ = self.classifier.fit(self.data_loader.train_X, self.data_loader.train_Y).predict_proba(X)
        fpr, tpr, thresholds = roc_curve(Y, probas_[:, 1])
        return fpr, tpr

    def get_auc(self,X,Y):
        """Gets the Area Under the Curve (AUC) score. This is area under the Receiver Operating 
        Characteristic curve, and is a measure of how balanced the classifier is in terms of false 
        negatives vs. false positives."""
        fpr, tpr = self.get_fpr_and_tpr(X,Y)
        return auc(fpr,tpr)

    def test_on_validation(self):
        """Returns performance on the model's validation set."""
        acc = self.get_accuracy(self.data_loader.val_X, self.data_loader.val_Y)
        auc = self.get_auc(self.data_loader.val_X, self.data_loader.val_Y)
        print "Final results on validation data:", acc, "accuracy", auc, "AUC"
        return acc, auc

    def test_on_test(self):
        """Returns performance on the model's test set."""
        print "WARNING! Only test on the test set when you have finished choosing all of your hyperparameters!"
        print "\tNever use the test set to choose hyperparameters!!!"
        
        acc = self.get_accuracy(self.data_loader.test_X, self.data_loader.test_Y)
        auc = self.get_auc(self.data_loader.test_X, self.data_loader.test_Y)
        print "Final results on test data!!", acc, "accuracy", auc, "AUC"
        return acc, auc
        
    def get_num_support_vectors(self):
        """Gets the number of support vectors used in forming the decision boundary."""
        return self.classifier.n_support_

    def get_hinge_loss(self,X,Y):
        """Computes hinge loss on the predictions made by the classifier on data points X compared
        to ground truth labels Y."""
        preds = self.predict(X)
        hinge_inner_func = 1.0 - preds*Y
        hinge_inner_func = [max(0,x) for x in hinge_inner_func]
        return sum(hinge_inner_func)

    def save_to_file(self, filepath):
        """Saves the trained classifier to a pickle file."""
        s = pickle.dumps(self.classifier)
        f = open(filepath, 'w')
        f.write(s)

    def load_from_file(self, filepath):
        """Loads a trained classifier from a pickle file. Warning: does not load the hyperparameters
        used to train the classifier into the main class."""
        f2 = open(filepath, 'r')
        s2 = f2.read()
        self.classifier = pickle.loads(s2)

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
            mesh_step_size = .02
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
