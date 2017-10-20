import pandas as pd
import numpy as np
from sklearn import tree

DATA_10_FOLD_DIR = '../data_10_fold'





K = 11

for k in range(1, K):
    
    train_feature_df = pd.read_csv(DATA_10_FOLD_DIR + '/train_feature_' + str(k) + '.csv').as_matrix()
    train_label_df = pd.read_csv(DATA_10_FOLD_DIR + '/train_label_' + str(k) + '.csv').as_matrix()
    test_feature_df = pd.read_csv(DATA_10_FOLD_DIR + '/test_feature_' + str(k) + '.csv').as_matrix()
    test_label_df = pd.read_csv(DATA_10_FOLD_DIR + '/test_label_' + str(k) + '.csv').as_matrix()
    
    #print(train_feature_df)
    #print(train_label_df)
    
    '''
    X = train_mat
    Y = train_mat_labels
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    '''





