from common_functions import *

# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

# Driver code

def dtree():
    # Building Phase
    data = importdata()
    X, y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

    count = 0
    acu = 0
    while count != 5:
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_pred = prediction(X_test,clf_entropy)
            acc = cal_accuracy(y_test, y_pred)
            acu += acc
        count += 1
    return acu / (count * 5)




