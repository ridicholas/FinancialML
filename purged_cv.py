import pandas as pd
import sklearn.metrics as skm


def makeFolds(X, y, numFolds: int, purgeBars: int):
    """
    :type X: pandas DataFrame
    :type y: pandas Series
    :type purgeBars: int
    :type numFolds: int
    """
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []
    numItems = int(X.shape[0] / numFolds)
    startDex = 0
    for i in range(1,numFolds+1):
        if startDex == 0:
            X_test = X.iloc[startDex:numItems-purgeBars]
            X_train = X.iloc[startDex+numItems:X.shape[0]]

        elif i == numFolds:
            X_test = X.iloc[startDex + purgeBars:X.shape[0]]
            X_train = X.iloc[0:startDex]

        else:
            X_test = X.iloc[startDex + purgeBars:startDex + numItems-purgeBars]
            X_train = pd.concat([X.iloc[0:startDex], X.iloc[startDex + numItems:X.shape[0]]])

        y_test = y.loc[X_test.index]
        y_train = y.loc[X_train.index]

        startDex += numItems
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    return pd.DataFrame({'X_train': pd.Series(X_trains),
                         'y_train': pd.Series(y_trains),
                         'X_test': pd.Series(X_tests),
                         'y_test': pd.Series(y_tests)})

def purgedF1CV(folds, model, classes, weights, tree_depths, max_leaves):
    results = pd.Series()
    for i in range(folds.shape[0]):
        model.fit(folds.X_train[i], folds.y_train[i])
        preds = model.predict(folds.X_test[i])
        results.append(skm.f1_score(folds.y_test[i], preds, labels=classes, average='weighted'))







