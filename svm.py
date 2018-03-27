from common_functions import *


def svm_algo():
    # Building Phase
    data = importdata()
    X, y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    # Operational Phase

    count=0
    acu=0
    while count!=5:
     kf = KFold(n_splits=5)
     for train_index, test_index in kf.split(X):
           X_train, X_test = X[train_index], X[test_index]
           y_train, y_test = y[train_index], y[test_index]
           y_pred = clf.predict(X_test)
           acc=cal_accuracy(y_test, y_pred)
           acu+=acc
     count+=1
    return acu / (count * 5)




