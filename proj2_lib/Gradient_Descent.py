import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import roc_curve, auc


class Gradient_Descent(object):
    
    def fit(self, X, Y):
        self.w, self.b = self.calculate_gradient_descent(X, Y)
        

    def calculate_gradient_descent(self, X, Y, num_iter=200, _lambda=1, learn_rate=0.1):
        b = 0
        w = np.full((X.shape[1], 1), 0.0)
        
        for k in range(0, num_iter):
            # Halve learning rate for every iteration
            learn_rate = learn_rate / 2
    
            #calculate gradient to w and b respectively
            gradient_to_w = self.calculate_gradient_to_w_using_hinge_loss(w, X, Y)
            '''TODO: have not considered regularizer'''        
            
            # Update b & w
            w = w - learn_rate * gradient_to_w
            print(k)
            #b = b - learn_rate * gradient_to_b
                
        return w, b



    def calculate_gradient_to_w_using_hinge_loss(self, w, X, Y):

        gradient_to_w = np.full((1, w.shape[0]), 0.0)
        for (x_, y_) in zip(X, Y):
            _x = np.full((1, x_.shape[0]), x_)
            
            '''TODO: add 2 * lambd * ||w||  here later at the end of f later'''
            f = _x.dot(w) 
            u = y_ * f
            
            # Using hinge loss ->gradient = -yx if yf < 1, or 0 if yf > 1
            if np.any(u < 1):
                gradient_curr = -y_ * x_
            else:
                gradient_curr = np.full((1, w.shape[0]), 0.0)    
                
            gradient_to_w += gradient_curr
            
            
        #print(gradient_to_w)
        return np.transpose(gradient_to_w)
    
    
    def predict(self, X):
        w = self.w
        b = self.b
        outcome_list = []
        
        print(type(X))
        
        for _x in X:
            x = np.full((1, _x.shape[0]), _x)
            f = x.dot(w) 
            #print(f)
            
            if f >= 0:
                outcome = 1.0
            else:
                outcome = -1.0
            outcome_list.append(outcome)
         
        return outcome_list    
            