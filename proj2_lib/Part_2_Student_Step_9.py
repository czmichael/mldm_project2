import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from Gradient_Descent import Gradient_Descent
from collections import Counter

DATA_10_FOLD_DIR = '../data_10_fold'

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
    gradient_descent = Gradient_Descent()
    gradient_descent.fit(X, Y)
    
    predicted = gradient_descent.predict(test_feature)
    actual = test_label.flatten()
  
    print('predicted -->', Counter(predicted))
    print('actual -->', Counter(actual))
    print('equals -->', sum(np.array(predicted)==np.array(actual)))
    
    accuracy = sum(np.array(predicted)==np.array(actual))/float(len(actual))
    fpr, tpr, thresholds = roc_curve(actual, predicted)
    roc_auc = auc(fpr, tpr)

    accuracy_total = accuracy_total + accuracy
    roc_auc_total = roc_auc_total + roc_auc
    

accuracy_final = accuracy_total / (K - 1)
roc_auc_final = roc_auc_total / (K - 1)

print ("for gradient descent, accuracy={}, AUC={}".format(accuracy_final, roc_auc_final))

