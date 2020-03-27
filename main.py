from TAQ import TAQ
import pandas as pd
import time
import matplotlib.pyplot as plt

def main():
    d = pd.read_csv('QQQ1day.csv')
    d = d.iloc[0:100, :]
    now = time.time()
    taq = TAQ(data=d)


    #print(taq.timeBars.head(10))
    #print(taq.data.head(20))
    #taq.candlePlot("QQQ", start = '2020-01-08 04:00:00', end = '2020-01-08 04:20:00', mav = 4, volume = True)
    #print(taq.timeBars.numTicks.mean())

    Ts, i_s, imbalances, abs_thetas, thresh = taq.identifyImbalanceIndexes('QQQ', 8)
    #Ts, abs_thetas, thresh, i_s = taq.compute_Ts(taq.data.ticks, 2000, abs(taq.data.ticks.mean()))
    print(Ts)
    print(i_s)
    #print(imbalances)
    print(thresh)
    print(abs_thetas)
    print('this took ' + str(time.time()-now) + ' seconds')

    plt.plot(abs_thetas)
    for num in i_s:
        plt.axvline(x=num)
    plt.show()
    plt.plot((taq.data.ticks * taq.data.volume).cumsum())
    for num in i_s:
        plt.axvline(x=num)

    plt.show()














if __name__ == '__main__':
    main()