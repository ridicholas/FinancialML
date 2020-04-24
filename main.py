from TAQ import TAQ
import pandas as pd
import time
from xgboost import XGBClassifier
import sklearn.metrics as skm
import math
import matplotlib.pyplot as plt
import purged_cv
def main():
    d = pd.read_csv('QQQ.csv')
    #d = d.iloc[0:2000]
    #d2 = d[::-1].reset_index(drop=True)
    #d = pd.concat([d,d2]).reset_index(drop=True)

    now = time.time()
    taq = TAQ(data=d, rm_outliers=False)
    #stuff = taq.data.ticks*taq.data.volume*taq.data.price
    #stuff.plot.kde()
    #plt.show()
    taq.makeGroup(taq.identifyRunsIndexes, ET_init = 10000, type='dollar', wordy = False, plotty=False, rm_outs = True)
    #print(taq.data.groups)
    #print(taq.timeBars.index)
    #print(taq.groupBars.index)
    #taq.candlePlot(mav=4, volume=True)
    #taq.candlePlot(type = 'group', mav=4, volume=True)
    print('this took ' + str(time.time() - now) + ' seconds')
    now=time.time()
    y_train = taq.tripleBarrierLabel(5, type='time')
    X_train = taq.timeBars
    X_train = X_train.drop(['sameDay', 'startTime', 'endTime'], axis=1)

    folds = purged_cv.makeFolds(X_train, y_train, 5, 5)
    purged_cv.purgedF1CV(folds, XGBClassifier(), 1)

    '''
    y_train = y_train.iloc[0:y_train.shape[0]-5]
    X_train = X_train.iloc[0:X_train.shape[0] - 5]
    
    X_test = X_train.iloc[200:234]
    y_test = y_train.iloc[200:234]
    X_train = X_train.iloc[0:195]
    y_train = y_train.iloc[0:195]

    model = XGBClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(sklearn.metrics.accuracy_score(y_test,y_pred))
    '''


    print('and this took: ' + str(time.time() - now) + ' seconds')






if __name__ == '__main__':
    main()