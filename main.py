from TAQ import TAQ
import pandas as pd
import time
import math
import matplotlib.pyplot as plt

def main():
    d = pd.read_csv('ibmaapl.csv')
    #d = d.iloc[0:350000]
    now = time.time()
    taq = TAQ(data=d, ticker='AAPL')
    taq.makeGroup(taq.identifyRunsIndexes, math.floor(taq.data.shape[0]*0.01), 'dollar')
    #print(taq.data.groups)
    #print(taq.timeBars.index)
    #print(taq.groupBars.index)
    #taq.candlePlot(mav=4, volume=True)
    #taq.candlePlot(type = 'group', mav=4, volume=True)
    print('this took ' + str(time.time() - now) + ' seconds')
    now=time.time()
    #taq.fixedTimeLabel(2)
    taq.tripleBarrierLabel(100)
    print('and this took: ' + str(time.time() - now) + ' seconds')






if __name__ == '__main__':
    main()