from common_functions import *

# Driver code

def mlp():
    data = importdata()
    X, y, X_train, X_test, y_train, y_test = splitdataset(data)
    mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
    mlp.fit(X_train, y_train)

    count=0
    acu=0
    while count!=5:
     kf = KFold(n_splits=5)
     for train_index, test_index in kf.split(X):
           X_train, X_test = X[train_index], X[test_index]
           y_train, y_test = y[train_index], y[test_index]
           y_pred = mlp.predict(X_test)
           acc=cal_accuracy(y_test, y_pred)
           acu+=acc
     count+=1
    return acu / (count * 5)


