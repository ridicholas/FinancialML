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
        self.Ts, self.abs_thetas, self.thresholds, self.i_s = self.compute_Ts(self.data.ticks,
                                                                         2000,
                                                                         abs(self.data.ticks.mean()))

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

    def compute_Ts(self, bvs, E_T_init, abs_Ebv_init):
        Ts, i_s = [], []
        i_prev, E_T, abs_Ebv = 0, E_T_init, abs_Ebv_init

        n = bvs.shape[0]
        bvs_val = bvs.values.astype(np.float64)
        abs_thetas, thresholds = np.zeros(n), np.zeros(n)
        abs_thetas[0], cur_theta = np.abs(bvs_val[0]), bvs_val[0]
        for i in range(1, n):
            cur_theta += bvs_val[i]
            abs_theta = np.abs(cur_theta)
            abs_thetas[i] = abs_theta

            threshold = E_T * abs_Ebv
            thresholds[i] = threshold

            if abs_theta >= threshold:
                cur_theta = 0
                Ts.append(np.float64(i - i_prev))
                i_s.append(i)
                i_prev = i
                E_T = _ewma(np.array(Ts), window=np.int64(len(Ts)))[-1]
                abs_Ebv = np.abs(_ewma(bvs_val[:i], window=np.int64(E_T_init * 3))[-1])  # window of 3 bars
        return Ts, abs_thetas, thresholds, i_s














