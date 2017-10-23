import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import roc_curve, auc

DATA_10_FOLD_DIR = '../data_10_fold'

def calculate_gradient_descent(X, Y, num_iter, _lambda, learn_rate):
    b = 0
    w = np.full((X.shape[1], 1), 3.0)
    print(w.shape)
    
    
    z = np.full((1, X.shape[1]), X[0])
    
    print(z.dot(w))

    
    
    for k in range(0, num_iter):
        # Halve learning rate for every iteration
        learn_rate = learn_rate / 2
        print(learn_rate)

        #calculate gradient to w and b respectively

        # Update b & w
        w = w - learn_rate * gradient_to_w
        b = b - learn_rate * gradient_to_b
            
    
    return w, b



K = 2
accuracy_total = 0
roc_auc_total = 0
for k in range(1, K):
    
    train_feature = pd.read_csv(DATA_10_FOLD_DIR + '/train_feature_' + str(k) + '.csv').as_matrix()
    train_label = pd.read_csv(DATA_10_FOLD_DIR + '/train_label_' + str(k) + '.csv').as_matrix()
    test_feature = pd.read_csv(DATA_10_FOLD_DIR + '/test_feature_' + str(k) + '.csv').as_matrix()
    test_label = pd.read_csv(DATA_10_FOLD_DIR + '/test_label_' + str(k) + '.csv').as_matrix()
    
    
    X = train_feature
    Y = train_label
    clf = calculate_gradient_descent(X, Y, 1500, 1, 0.1)
    
  
    '''  
    predicted = clf.predict(test_feature)
    actual = test_label.flatten()
    
    accuracy = sum(np.array(predicted)==np.array(actual))/float(len(actual))
    fpr, tpr, thresholds = roc_curve(actual, predicted)
    roc_auc = auc(fpr, tpr)
    #print("K={}, accuracy={}, AUC={}".format(k, accuracy, roc_auc))

    accuracy_total = accuracy_total + accuracy
    roc_auc_total = roc_auc_total + roc_auc


accuracy_final = accuracy_total / 10
roc_auc_final = roc_auc_total / 10

print ("for decision tree, accuracy={}, AUC={}".format(accuracy_final, roc_auc_final))
''' 
