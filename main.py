from TAQ import TAQ
import pandas as pd
import time
import matplotlib.pyplot as plt

def main():
    d = pd.read_csv('QQQ1day.csv')
    d = d.iloc[0:10000, :]
    now = time.time()
    taq = TAQ(data=d)


    #print(taq.timeBars.head(10))
    #print(taq.data.head(20))
    #taq.candlePlot("QQQ", start = '2020-01-08 04:00:00', end = '2020-01-08 04:20:00', mav = 4, volume = True)
    #print(taq.timeBars.numTicks.mean())

    Ts, i_s, imbalances, abs_thetas, thresh, thetas = taq.identifyImbalanceIndexes('QQQ', 2000)
    print(Ts)
    print(i_s)
    print(imbalances)
    print(thresh)
    print(abs_thetas)
    print('this took ' + str(time.time()-now) + ' seconds')

    plt.plot(abs_thetas)
    for num in i_s:
        plt.axvline(x=num)
    plt.show()
    plt.plot(thetas)
    for num in i_s:
        plt.axvline(x=num)
    plt.show()
    plt.plot(taq.data.ticks * taq.data.price * taq.data.volume)
    plt.show()

    superTicks = taq.data.ticks * taq.data.price * taq.data.volume
    print(superTicks[superTicks == max(superTicks)])
    #abs_thetas.to_csv('abs_thetas.csv')
    #thetas.to_csv('thetas.csv')















if __name__ == '__main__':
    main()