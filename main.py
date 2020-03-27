from TAQ import TAQ
import pandas as pd
import time

def main():
    d = pd.read_csv('QQQ1day.csv')
    d = d.iloc[0:1000, :]
    now = time.time()
    taq = TAQ(data=d)
    print(taq.thresholds[0:50])
    print(taq.Ts)
    print(taq.i_s[0:50])
    print("this took " + str(time.time() - now) + " seconds")

    #print(taq.timeBars.head(10))
    #print(taq.data.head(20))
    #taq.candlePlot("QQQ", start = '2020-01-08 04:00:00', end = '2020-01-08 04:20:00', mav = 4, volume = True)
    #print(taq.timeBars.numTicks.mean())
    #Ts, abs_thetas, thresholds, i_s = taq.compute_Ts(taq.data.ticks, 200, abs(taq.data.ticks.mean()))




if __name__ == '__main__':
    main()