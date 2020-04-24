import numpy as np
import pandas as pd
import sklearn
import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy import stats
import random


class TAQ():
    """TAQ object generated from WRDS database TAQ trade csv file.
    Stores initial CSV as pd dataframe and renames columns"""

    def __init__(self, path='null', data=None, ticker=None, rm_outliers=False):

        if path != 'null':
            self.taqPath = path
            self.rawData = pd.read_csv(path)
        elif data is not None:
            self.rawData = data

        if ticker == None:
            first = self.rawData.SYM_ROOT[0]
            self.rawData = self.rawData[self.rawData.SYM_ROOT == first]
        else:
            self.rawData = self.rawData[self.rawData.SYM_ROOT == ticker].reset_index(drop=True)

        self.data = self.preprocess()
        self.rm_outliers = rm_outliers
        if self.rm_outliers:
            self.data = self.data[self.data.volume < self.data.volume.quantile(.999)].reset_index(drop=True)

        self.timeBars = self.makeBars()
        self.groupBars = None

    def makeGroup(self, func, ET_init=0, type='dollar', wordy=False, plotty=False, rm_outs=False):
        Ts, i_s, imbalances, abs_thetas, thresh, thetas, ticks = func(ET_init, type, rm_outs)

        # show results
        if wordy:
            print(Ts)
            print(i_s)
            print(imbalances)
            print(thresh)
            print(abs_thetas)

        # plot results
        if plotty:
            plt.plot(abs_thetas)
            for num in i_s:
                plt.axvline(x=num, alpha=0.2, color='gray')
            plt.title('Final Expected Runs')
            plt.ylabel('Cumulative Dollars')
            plt.xlabel('Index')
            plt.show()
            plt.plot((self.data.ticks * self.data.volume * self.data.price).cumsum())
            plt.title('Dollar Cumulative Sum Over Time')
            plt.ylabel('Cumulative Dollars')
            plt.xlabel('Index')
            for num in i_s:
                plt.axvline(x=num, alpha=0.2, color='gray')
            plt.show()
            plt.plot((self.data.ticks * self.data.volume * self.data.price).cumsum())
            plt.title('Dollar Cumulative Sum Over Time')
            plt.ylabel('Cumulative Dollars')
            plt.xlabel('Index')
            plt.show()

        prev_i = 0
        group = 0
        for i in i_s:
            if i == 0:
                continue
            else:
                self.data.groups[prev_i:i] = group
                prev_i = i
                group += 1
        self.groupBars = self.makeBars(group=True)

    def make_timestamp(self):
        combined = self.rawData['DATE'].apply(str) + ' ' + self.rawData['TIME_M']
        timestamp = pd.to_datetime(combined)
        return timestamp

    def preprocess(self):
        data = self.rawData[['SYM_ROOT', 'EX', 'SIZE', 'PRICE']]
        data = data.rename(columns={"SYM_ROOT": "ticker", "EX": 'exchange', "SIZE": 'volume', "PRICE": 'price'})
        data['priceXvolume'] = data['price'] * data['volume']
        data['timestamp'] = self.make_timestamp()
        data['date'] = data['timestamp'].dt.date
        ticks = data['price'].diff() / abs(data['price'].diff())
        newSym = data['ticker'].ne(data['ticker'].shift(1))
        ticks[newSym] = 0
        ticks = ticks.fillna(method='ffill')
        data['ticks'] = ticks
        data = data[data.ticks != 0]
        data['groups'] = np.zeros(data.shape[0])
        return data.reset_index(drop=True)

    def makeBars(self, group=False, timelevel='min'):
        """Will create TimeBars using level specified in pre-processing"""
        groupData = self.data.copy(deep=True)
        groupData['oTimestamp'] = groupData.timestamp.copy(deep=True)
        if group == True:
            if self.data.groups.sum() > 0:
                groupData = groupData.groupby(['ticker', 'groups'])
            else:
                print('there are no groups to groupby')
        else:
            groupData.timestamp = groupData.timestamp.dt.floor(timelevel)
            groupData = groupData.groupby(['ticker', 'timestamp'])
        volsum = groupData['volume'].sum()
        psum = groupData['priceXvolume'].sum()
        vwap = psum / volsum
        o = groupData['price'].first()
        c = groupData['price'].last()
        high = groupData['price'].max()
        low = groupData['price'].min()
        startTime = groupData['oTimestamp'].first()
        endTime = groupData['oTimestamp'].last()
        frame = {'Volume': volsum,
                 'VWAP': vwap,
                 'Open': o,
                 'Close': c,
                 'High': high,
                 'Low': low,
                 'numTicks': groupData['price'].count(),
                 'startTime': startTime,
                 'endTime': endTime,
                 'elapsedTime': (endTime - startTime).dt.total_seconds(),
                 'startDayOfWeek': startTime.dt.dayofweek,
                 'sameDay': endTime.dt.dayofweek == startTime.dt.dayofweek}

        frame = pd.DataFrame(frame)
        if group == True:
            arrays = [frame.index.get_level_values(0), frame.startTime]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=['ticker', 'timestamp'])
            frame.index = index
        return frame

    def candlePlot(self, type='time', mav=None, start=None, end=None, volume=False):
        plt.figure(figsize=(45, 15))
        if type == 'group':
            plotData = self.groupBars.droplevel(axis=0, level=0)
        else:
            plotData = self.timeBars.droplevel(axis=0, level=0)
        if start and end:
            plotData = plotData.loc[start:end, :]
        if mav:
            mpf.plot(plotData, type='candle', mav=mav, volume=volume)
        else:
            mpf.plot(plotData, type='candle', volume=volume)
        plt.show()

    def linePlot(self, type='time', start=None, end=None):
        plt.figure(figsize=(45, 15))
        if type == 'group':
            plotData = self.groupBars.droplevel(axis=0, level=0)
        else:
            plotData = self.timeBars.droplevel(axis=0, level=0)
        if start and end:
            plotData = plotData.loc[start:end, :]

        mpf.plot(plotData, type='line')
        plt.show()

    def identifyRunsIndexes(self, ET_init=0, type='dollar', rm_outs=True):
        if ET_init == 0:
            ET_init = self.data.shape[0] * 0.01

        ET = ET_init
        ticker = self.data.copy(deep=True)

        if type == 'tick':
            ticks = ticker.ticks
        elif type == 'volume':
            ticks = ticker.ticks * ticker.volume
        else:
            ticks = ticker.ticks * ticker.volume * ticker.price

        n = ticks.shape[0]
        outliers = ticks[abs(ticks) >= abs(ticks).quantile(n/(n+5))]
        if rm_outs:
            ticks = ticks[abs(ticks) < abs(ticks).quantile(n/(n+5))].reset_index(drop=True)
        posticks = ticks.copy(deep=True)
        posticks[posticks < 0] = 0
        negticks = ticks.copy(deep=True)
        negticks[negticks > 0] = 0
        df = pd.DataFrame({"pos": posticks.cumsum(), "neg": -negticks.cumsum()})
        thetas = df[["pos", "neg"]].max(axis=1)
        abs_thetas = abs(thetas)
        Eimbalance = max(posticks.mean(), abs(negticks.mean()))
        final_abs_thetas = abs_thetas.copy(deep=True)
        T = []
        i_s = [0]
        imbalances = []
        thresholds = []
        startThreshold = ET * Eimbalance

        i = 0

        while i < n:

            if len(T) >= 3:
                barBools = (abs_thetas >= pd.Series(thresholds[-3:-1]).ewm(com=0.5).mean().iloc[-1])

            else:
                barBools = (abs_thetas >= startThreshold)
            if sum(barBools) == 0:
                T.append(i - i_s[-1])
                i_s.append(n)
                imbalances.append(max(posticks[i_s[-1]:i].mean(), abs(negticks[i_s[-1]:i].mean())))
                break

            if len(imbalances) > 1 and max(posticks[i_s[-1]:barBools[barBools == True].index[0]].mean(),
                   abs(negticks[i_s[-1]:barBools[barBools == True].index[0]].mean())) > imbalances[-1]:
                i = barBools[barBools == True].index[0]
            else:
                i = barBools[barBools == True].index[0] + 1


            imbalances.append(max(posticks[i_s[-1]:i].mean(), abs(negticks[i_s[-1]:i].mean())))
            T.append(i - i_s[-1])

            df.pos[i_s[-1]:i - 1] = 0
            df.neg[i_s[-1]:i - 1] = 0
            df.pos[i - 1:] = df.pos[i - 1:] - df.pos[i - 1]
            df.neg[i - 1:] = df.neg[i - 1:] - df.neg[i - 1]

            abs_thetas = df[["pos", "neg"]].max(axis=1)

            final_abs_thetas[i:] = abs_thetas[i:]
            i_s.append(i)

            thresholds.append(T[-1] * imbalances[-1])
            # print("T: " + str(T[-1]) + ", imbalance: " + str(imbalances[-1])+ ", Threshold: " + str(thresholds[-1]))

        i_s = pd.Series(i_s)
        if rm_outs:
            for index in outliers.index:
                i_s[i_s > index] = i_s[i_s > index] + 1

        return T, i_s, imbalances, final_abs_thetas, thresholds, thetas, ticks

    def fixedTimeLabel(self, numBars, type='time'):
        if type == 'group':
            returns = self.groupBars.VWAP
        else:
            returns = self.timeBars.VWAP
        print(returns)

        returns = -returns.diff(periods=-numBars) / returns
        tau = returns.std()

        returns[(returns > -tau) & (returns < tau)] = 0
        returns[returns > tau] = 1
        returns[returns < -tau] = -1

        return returns

    def tripleBarrierLabel(self, numBars, upper=2, lower=2, type='time'):

        if type == 'group':
            returns = self.groupBars.VWAP.copy(deep=True)
            highs = self.groupBars.High.copy(deep=True)
            lows = self.groupBars.Low.copy(deep=True)
        else:
            returns = self.timeBars.VWAP.copy(deep=True)
            highs = self.timeBars.High.copy(deep=True)
            lows = self.timeBars.Low.copy(deep=True)

        returns = pd.DataFrame(returns)
        highs = pd.DataFrame(highs)
        lows = pd.DataFrame(lows)
        for i in range(1, numBars + 1):
            label = 'shifted: ' + str(i)
            returns[label] = -returns.VWAP.diff(periods=-i) / returns.VWAP
            highs[label] = (highs.High.shift(periods=-i) - returns.VWAP) / returns.VWAP
            lows[label] = (lows.Low.shift(periods=-i) - returns.VWAP) / returns.VWAP

        tau = returns.drop('VWAP', axis=1).std()[-1]
        shifts = returns.copy(deep=True)
        shifts = shifts.drop('VWAP', axis=1)
        highs = highs.drop('High', axis=1)
        lows = lows.drop('Low', axis=1)
        shifts = (highs.gt(upper * tau, axis=1)) | (lows.lt(-lower * tau, axis=1))
        shifts[label] = True
        shifts = shifts.cumsum(axis=1)
        shifts = shifts.cumsum(axis=1)
        returns[shifts != 1] = 0
        returns.VWAP = 0
        label = returns.sum(axis=1)
        label[(label > -tau) & (label < tau)] = 0
        label[label > tau] = 1
        label[label < -tau] = -1

        return label