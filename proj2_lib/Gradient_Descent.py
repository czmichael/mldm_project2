import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from numpy import linalg as LA


class Gradient_Descent(object):
    
    def fit(self, X, Y):
        self.w, self.b = self.calculate_gradient_descent(X, Y)
        

    def calculate_gradient_descent(self, X, Y, num_iter=500, _lambda=1, learn_rate=0.1):
        b = 0
        w = np.full((X.shape[1], 1), 0.0)
        
        for k in range(0, num_iter):
            print('current iteration -> {}'.format(k))
            
            # Halve learning rate for every iteration
            learn_rate = learn_rate / 2
    
            #calculate gradient to w and b respectively
            gradient_to_w, gradient_to_b = self.calculate_gradient_using_hinge_loss(w, X, Y, _lambda, b)
            
            # Update b & w
            w = w - learn_rate * gradient_to_w
            b = b - learn_rate * gradient_to_b
                
        return w, b



    def calculate_gradient_using_hinge_loss(self, w, X, Y, _lambda, b):

        gradient_to_w = np.full((1, w.shape[0]), 0.0)
        gradient_to_b = 0
        
        for (x_, y_) in zip(X, Y):
            _x = np.full((1, x_.shape[0]), x_)
            
            f = _x.dot(w) + b
            u = y_ * f
            
            # Using hinge loss ->gradient = -yx if yf < 1, or 0 if yf > 1
            if np.any(u < 1):
                gradient_w_curr = -y_ * x_ + (2 * _lambda * LA.norm(w))
                gradient_b_curr = -y_
            else:
                gradient_w_curr = np.full((1, w.shape[0]), 0.0)   
                gradient_b_curr = 0 
                
            gradient_to_w += gradient_w_curr
            gradient_to_b += gradient_b_curr
            
        #print(gradient_to_w)
        return np.transpose(gradient_to_w), gradient_to_b
    
    
    def predict(self, X):
        w = self.w
        b = self.b
        outcome_list = []
        
        print(type(X))
        
        for _x in X:
            x = np.full((1, _x.shape[0]), _x)
            f = x.dot(w) + b
            
            if f >= 0:
                outcome = 1.0
            else:
                outcome = -1.0
            outcome_list.append(outcome)
         
        return outcome_list    
            