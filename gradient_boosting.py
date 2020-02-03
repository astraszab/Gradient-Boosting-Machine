#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

class LS_Boost:
    """Gradient Boosting for regression optimizing MSE.
    
    
    Keyword argumetns:
    num_trees -- number of boosting stages (positive int, default 100)
    max_depth -- maximal depth of each tree (positive int, default 3)
    learning_rate -- coeffitient of contrubution for each tree (positive float, default 1.0)
    
    More information: https://statweb.stanford.edu/~jhf/ftp/trebst.pdf
    """
    def __init__(self, num_trees=100, max_depth=3, learning_rate=1.0):
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.train_mean = None
        self.fitted = False
        
    def fit(self, X, y):
        """Fit the gradient boosting model and return it.
        
        Keyword arguments:
        X -- the input samples (array-like of real numbers of shape (n_samples, n_features))
        y -- target values (array-like of real numbers of shape (n_samples, ))
        """
        self.train_mean = y.mean()
        F = self.train_mean * np.ones(len(y))
        for _ in range(self.num_trees):
            gradients = y - F
            decision_tree = DecisionTreeRegressor(max_depth=self.max_depth)
            decision_tree.fit(X, gradients)
            self.trees.append(decision_tree)
            F += self.learning_rate * decision_tree.predict(X)
        self.fitted = True
        return self
            
    def predict(self, X):
        """Predict regression target for X. Return pandas series of predictions.
        
        Keyword arguments:
        X -- the input samples (array-like of real numbers of shape (n_samples, n_features))
        """
        if not self.fitted:
            raise Exception('Model is not fit.')
        preds = self.train_mean * np.ones(len(X))
        for decision_tree in self.trees:
            preds += self.learning_rate * decision_tree.predict(X)
        return pd.Series(data=preds, index=X.index)
    

class L2TreeBoost:
    """Gradient Boosting for binary classification optimizing binary cross-entropy.
    
    
    Keyword argumetns:
    num_trees -- number of boosting stages (positive int, default 100)
    max_depth -- maximal depth of each tree (positive int, default 3)
    learning_rate -- coeffitient of contrubution for each tree (positive float, default 1.0)
    
    More information: https://statweb.stanford.edu/~jhf/ftp/trebst.pdf
    """
    def __init__(self, num_trees=100, max_depth=3, learning_rate=1.0):
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.epsilon = 0.00001
        self.trees = []
        self.F0 = None
        self.fitted = False
        
    def fit(self, X, y):
        """Fit the gradient boosting model and return it.
        
        Keyword arguments:
        X -- the input samples (array-like of real numbers of shape (n_samples, n_features))
        y -- target values (array-like of classes {0, 1} of shape (n_samples, ))
        """
        self.gamma = np.zeros((self.num_trees, 2**(self.max_depth + 1) - 1))
        y = y * 2 - 1
        self.F0 = np.log((1 + y.mean())/(1 - y.mean()))/2
        F = self.F0 * np.ones(len(y))
        for k in range(self.num_trees):
            gradients = 2 * y / (1 + np.exp(2 * y * F))
            decision_tree = DecisionTreeRegressor(max_depth=self.max_depth)
            decision_tree.fit(X, gradients)
            self.trees.append(decision_tree)
            leaves = decision_tree.apply(X)
            for leaf in np.unique(leaves):
                leaf_gradients = gradients[leaves == leaf]
                self.gamma[k, leaf] = (np.sum(leaf_gradients) 
                                  / (np.dot(np.abs(leaf_gradients), 2 - np.abs(leaf_gradients)) 
                                     + self.epsilon))
            F += self.learning_rate * self.gamma[k, leaves]
        self.fitted = True
        return self
        
    def predict_proba(self, X):
        """Predict probabilities that the classification target is 1 for X. Return pandas series of probabilities.
        
        Keyword arguments:
        X -- the input samples (array-like of real numbers of shape (n_samples, n_features))
        """
        if not self.fitted:
            raise Exception('Model is not fit.')
        F = self.F0 * np.ones(len(X))
        for k, decision_tree in enumerate(self.trees):
            leaves = decision_tree.apply(X)
            F += self.learning_rate * self.gamma[k, leaves]
        proba = 1 / (1 + np.exp(-2*F))
        return pd.Series(data=proba, index=X.index)
            
    def predict(self, X):
        """Predict classification target for X. Return pandas series of predictions from {0, 1}.
        
        Keyword arguments:
        X -- the input samples (array-like of real numbers of shape (n_samples, n_features))
        """
        if not self.fitted:
            raise Exception('Model is not fit.')
        return self.predict_proba(X) > 0.5