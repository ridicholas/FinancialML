import pandas as pd
import sklearn.metrics as skm
import xgboost
import time
import sklearn


def makeFolds(X, y, numFolds: int, purgeBars: int):
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []
    weights = []
    numItems = int(X.shape[0] / numFolds)
    startDex = 0
    for i in range(1, numFolds + 1):
        if startDex == 0:
            X_test = X.iloc[startDex:numItems - purgeBars]
            X_train = X.iloc[startDex + numItems:X.shape[0]]

        elif i == numFolds:
            X_test = X.iloc[startDex + purgeBars:X.shape[0]]
            X_train = X.iloc[0:startDex]

        else:
            X_test = X.iloc[startDex + purgeBars:startDex + numItems - purgeBars]
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


def purgedPrecisionCV(folds, classes, weights, min_child_weights, max_depths, gammas, colsample_bytrees, eval_metrics,
               learning_rates, n_estimators):
    best_result = 0
    best_model = 0
    iter = 1
    numIters = len(weights) * len(min_child_weights) * len(max_depths) * len(gammas) * len(colsample_bytrees) * len(
        eval_metrics) * len(learning_rates) * len(n_estimators)
    final_weights = 0
    for weight in weights:
        for min_child_weight in min_child_weights:
            for max_depth in max_depths:
                for gamma in gammas:
                    for learning_rate in learning_rates:
                        for n_estimator in n_estimators:
                            for colsample_bytree in colsample_bytrees:
                                for eval_metric in eval_metrics:
                                    now = time.time()
                                    model = xgboost.XGBClassifier(min_child_weight=min_child_weight,
                                                                  max_depth=max_depth,
                                                                  gamma=gamma,
                                                                  colsample_bytree=colsample_bytree,
                                                                  eval_metric=eval_metric,
                                                                  n_estimators=n_estimator,
                                                                  learning_rate=learning_rate)
                            results = []

                            for i in range(folds.shape[0]):
                                model.fit(folds.X_train[i], folds.y_train[i],
                                          sample_weight=folds.y_train[i].replace(to_replace=[-1, 0, 1], value=weight))
                                preds = model.predict(folds.X_test[i])
                                results.append(skm.precision_score(folds.y_test[i], preds, labels=classes, average='micro'))
                            result = pd.Series(results).mean()
                            print(
                                "iteration {}/{} took {} seconds, result: {}".format(iter, numIters, time.time() - now,
                                                                                     result))
                            iter += 1
                            if result > best_result:
                                best_result = result
                                best_model = model
                                final_weights = weight
    return best_result, best_model, final_weights
