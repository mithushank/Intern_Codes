from keras.datasets import cifar10
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from RandomForest import RandomForest
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


n_tree_list = [2]
e_time = []
acc_list = []
#timer Start
for n in n_tree_list:
    start_time = time.time()

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    Reshaped = 3072
    X_train = X_train.reshape(50000, Reshaped)
    X_test = X_test.reshape(10000, Reshaped)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    #normalize the datast
    X_train/=255.
    X_test/=255.
    print(X_train[0])
    df = pd.DataFrame()
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0],'test samples')
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    clf = RandomForest(n_trees=n)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    #timer end
    end_time = time.time()
    elapsed_time = end_time - start_time
    e_time.append(elapsed_time)

    acc =  accuracy(y_test, predictions)
    acc_list.append(acc)
    print(acc)
    print(f"Time taken: {elapsed_time} seconds")
