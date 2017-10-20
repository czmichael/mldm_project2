import pandas as pd
import numpy as np
from sklearn import tree

DATA_10_FOLD_DIR = '../data_10_fold'

PROCESSED_DATA_DIR = '../processed_data'
CSV_FILE = PROCESSED_DATA_DIR + '/appt_mat.csv'

appt_df = pd.read_csv(CSV_FILE)
print(appt_df.head())

df_feature = appt_df.iloc[:, :101].as_matrix()
df_label = appt_df.iloc[:, 101:].as_matrix()




K = 11


for k in range(1, K):
    
    train_feature = []
    train_label = []
    test_feature = []
    test_label = []
    
    for n in range(1, df_feature.shape[0]):
        if n % K != k - 1:
            train_feature.append(df_feature[n].tolist())
            train_label.append(df_label[n].tolist())

        else:
            test_feature.append(df_feature[n].tolist())
            test_label.append(df_label[n].tolist())
            #test_mat.append(n)
    #print(train_mat)
    #print(train_label[:5])
    #print(test_mat)    
    #print(train_mat)
    #print("----- train shape: {}".format(np.shape(train_mat)))
    #print("----- test shape: {}".format(np.shape(test_mat)))   
    train_feature_df = pd.DataFrame(train_feature)
    train_label_df = pd.DataFrame(train_label)
    test_feature_df = pd.DataFrame(test_feature)
    test_label_df = pd.DataFrame(test_label)
    
    
    train_feature_df.to_csv(DATA_10_FOLD_DIR + '/train_feature_' + str(k) + '.csv', index=False)
    train_label_df.to_csv(DATA_10_FOLD_DIR + '/train_label_' + str(k) + '.csv', index=False)
    test_feature_df.to_csv(DATA_10_FOLD_DIR + '/test_feature_' + str(k) + '.csv', index=False)
    test_label_df.to_csv(DATA_10_FOLD_DIR + '/test_label_' + str(k) + '.csv', index=False)    
    
    
    
    '''
    X = train_mat
    Y = train_mat_labels
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    '''





