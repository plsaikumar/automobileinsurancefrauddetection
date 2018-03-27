import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold


# Function importing Dataset
def importdata():
    balance_data = pd.read_csv( 'result.csv',sep=',', header=None)
    pd.get_dummies(balance_data)
    return balance_data


# Function to split the dataset
def splitdataset(balance_data):
    X = balance_data.values[:, 1:18]
    Y = balance_data.values[:, 19]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    return X, Y, X_train, X_test, y_train, y_test



# Function to make predictions
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred) * 100
    #print("Accuracy : ",acc)
    #print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    return acc

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem