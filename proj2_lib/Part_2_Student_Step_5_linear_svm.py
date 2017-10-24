import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc

DATA_10_FOLD_DIR = '../data_10_fold'

    
K = 11
accuracy_total = 0
roc_auc_total = 0
for k in range(1, K):
    
    train_feature = pd.read_csv(DATA_10_FOLD_DIR + '/train_feature_' + str(k) + '.csv').as_matrix()
    train_label = pd.read_csv(DATA_10_FOLD_DIR + '/train_label_' + str(k) + '.csv').as_matrix()
    test_feature = pd.read_csv(DATA_10_FOLD_DIR + '/test_feature_' + str(k) + '.csv').as_matrix()
    test_label = pd.read_csv(DATA_10_FOLD_DIR + '/test_label_' + str(k) + '.csv').as_matrix()
    
    X = train_feature
    Y = train_label
    clf = LinearSVC(random_state=0)
    clf = clf.fit(X, Y)
    
    predicted = clf.predict(test_feature)
    actual = test_label.flatten()
    
    accuracy = sum(np.array(predicted)==np.array(actual))/float(len(actual))
    fpr, tpr, thresholds = roc_curve(actual, predicted)
    roc_auc = auc(fpr, tpr)
    print("K={}, accuracy={}, AUC={}".format(k, accuracy, roc_auc))

    accuracy_total = accuracy_total + accuracy
    roc_auc_total = roc_auc_total + roc_auc


accuracy_final = accuracy_total / 10
roc_auc_final = roc_auc_total / 10

print ("for Linear SVM, accuracy={}, AUC={}".format(accuracy_final, roc_auc_final))
