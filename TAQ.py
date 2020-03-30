import numpy as np
import pandas as pd
import sklearn
import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
from numba import jit
from numba import float64
from numba import int64
from fast_ewma import _ewma

class TAQ():
    """TAQ object generated from WRDS database TAQ trade csv file.
    Stores initial CSV as pd dataframe and renames columns"""

    def __init__(self, path='null', data=None):
        if path != 'null':
            self.taqPath = path
            self.rawData = pd.read_csv(path)
        elif data is not None:
            self.rawData = data

        self.data = self.preprocess()
        self.timeBars = self.makeTimeBars()

    def make_timestamp(self, level = 'Min'):
        combined = self.rawData['DATE'].apply(str) + ' ' + self.rawData['TIME_M']
        timestamp = pd.to_datetime(combined)
        return timestamp.dt.floor(level)

    def preprocess(self, level = 'Min'):
        data = self.rawData[['SYM_ROOT', 'EX', 'SIZE', 'PRICE']]
        data = data.rename(columns={"SYM_ROOT": "ticker", "EX": 'exchange', "SIZE" : 'volume', "PRICE": 'price'})
        data['priceXvolume'] = data['price'] * data['volume']
        data['timestamp'] = self.make_timestamp(level)
        data['date'] = data['timestamp'].dt.date
        ticks = data['price'].diff()/abs(data['price'].diff())
        newSym = data['ticker'].ne(data['ticker'].shift(1))
        ticks[newSym] = 0
        ticks = ticks.fillna(method = 'ffill')
        data['ticks'] = ticks
        return data



    def makeTimeBars(self):
        """Will create TimeBars using level specified in pre-processing"""

        groupData = self.data.groupby(['ticker', 'timestamp'])
        volsum = groupData['volume'].sum()
        psum = groupData['priceXvolume'].sum()
        vwap = psum / volsum
        o = groupData['price'].first()
        c = groupData['price'].last()
        high = groupData['price'].max()
        low = groupData['price'].min()
        frame = {'Volume': volsum, 'VWAP': vwap, 'Open': o, 'Close': c, 'High': high, 'Low': low, 'numTicks': groupData['price'].count()}
        frame = pd.DataFrame(frame)
        return frame

    def candlePlot(self, symbol, mav = None, start = None, end = None, volume = False):
        plt.figure(figsize=(45,15))
        plotData = self.timeBars.loc[symbol]
        if start and end:
            plotData = plotData.loc[start:end, :]
        if mav:
            mpf.plot(plotData, type = 'candle', mav = mav, volume = volume)
        else:
            mpf.plot(plotData, type='candle', volume = volume)
        plt.show()

    def identifyImbalanceIndexes(self, ticker, ET_init):
        ET = ET_init
        ticker = self.data[self.data.ticker == ticker]
        ticks = ticker.ticks * ticker.volume * ticker.price

        Eimbalance = abs(ticks.mean())
        thetas = ticks.cumsum()
        abs_thetas = np.abs(thetas)
        final_abs_thetas = abs_thetas.copy(deep=True)
        n = ticks.shape[0]

        T = []
        i_s = [0]
        imbalances = []
        thresholds = []
        startThreshold = ET*Eimbalance
        thresholds.append(startThreshold)

        i = 0
        while i < n:
            barBools = abs_thetas < thresholds[-1]
            if sum(barBools) == n:
                break
            i = barBools[barBools == False].index[0]
            imbalances.append(abs(ticks[i_s[-1]:i].mean()))
            if len(T) > 0:
                T.append(i - i_s[-1])
            else:
                T.append(i)

            abs_thetas[i_s[-1]:i] = 0

            abs_thetas[i:] = thetas[i:] - thetas[i]

            abs_thetas = abs(abs_thetas)
            final_abs_thetas[i:] = abs_thetas[i:]
            i_s.append(i)
            ET = pd.Series(T[max(-3, -len(T)):]).ewm(com=0.8).mean().iloc[-1]
            Eimbalance = pd.Series(imbalances[max(-3, -len(imbalances)):]).ewm(com=0.8).mean().iloc[-1]

            thresholds.append(ET * Eimbalance)





        return T, i_s, imbalances, final_abs_thetas, thresholds, thetas























