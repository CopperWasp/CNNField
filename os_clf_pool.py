import os_clf as clf
import numpy as np
from sklearn.linear_model import SGDClassifier

# SGD
img_dim = 230400

class pool:
    def __init__(self):
        self.classifiers = {}
        self.errors = {}
        self.occurrences = {}
        
        
    def add_classifier(self, new_class, index):
        #self.classifiers[new_class] = clf.olsf()
        #
        self.classifiers[new_class] = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, tol=-5, average=10) # sgd
        self.classifiers[new_class].partial_fit(np.ones(img_dim).reshape(1,-1), [1], classes=[-1,1]) # initialization
        #
        self.errors[new_class] = 1.0
        self.occurrences[new_class] = index
        
        
    def predict(self, row):
        self.predictions = {}
        for key in self.classifiers.keys():
            c = self.classifiers[key]
            #result = c.predict(row)
            result = c.predict(row.reshape(1,-1))[0] # sgd
            self.predictions[key] = result
        return self.predictions
    
    
    def expand(self, y, index):
        for label in y.keys():
            if label not in self.predictions.keys():
                self.add_classifier(label, index)           
        
        
    def fit(self, row, y, index):
        for key in self.classifiers.keys():
            c = self.classifiers[key]
            y_hat = np.sign(self.predictions[key])
            if key in y.keys():
                #is_false = c.fit(row, y_hat, y[key])
                c.partial_fit(row.reshape(1,-1), [np.sign(y[key])]) # sgd
                self.errors[key] += (y_hat == np.sign(y[key])) # sgd
            else:
                #is_false = c.fit(row, y_hat, -1) # sgd
                c.partial_fit(row.reshape(1,-1), [-1]) # sgd  
                self.errors[key] += (y_hat == -1) # sgd


        self.expand(y, index)

        
        