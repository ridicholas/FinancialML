import numpy as np
import pandas as pd
import sklearn
import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf



class TAQ():
    """TAQ object generated from WRDS database TAQ trade csv file.
    Stores initial CSV as pd dataframe and renames columns"""

    def __init__(self, path='null', data=None, ticker = None):

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
        self.timeBars = self.makeBars()
        self.groupBars = None

    def makeGroup(self, func, ET_init, type = 'dollar', wordy = False, plotty = False):
        Ts, i_s, imbalances, abs_thetas, thresh, thetas, ticks = func(ET_init, type)

        #show results
        if wordy:
            print(Ts)
            print(i_s)
            print(imbalances)
            print(thresh)
            print(abs_thetas)

        #plot results
        if plotty:
            plt.plot(abs_thetas)
            for num in i_s:
                plt.axvline(x=num, alpha = 0.2, color = 'gray')
            plt.title('Final Expected Runs')
            plt.ylabel('Cumulative Dollars')
            plt.xlabel('Index')
            plt.show()
            plt.plot((self.data.ticks * self.data.volume * self.data.price).cumsum())
            plt.title('Dollar Cumulative Sum Over Time')
            plt.ylabel('Cumulative Dollars')
            plt.xlabel('Index')
            for num in i_s:
                plt.axvline(x=num, alpha = 0.2, color = 'gray')
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
                group+=1
        self.groupBars = self.makeBars(group=True)


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
        data = data[data.ticks != 0]
        data['groups'] = np.zeros(data.shape[0])
        return data.reset_index(drop=True)



    def makeBars(self, group = False):
        """Will create TimeBars using level specified in pre-processing"""
        if group == True:
            if self.data.groups.sum() > 0:
                groupData = self.data.groupby(['ticker', 'groups'])
            else:
                print('there are no groups to groupby')
        else:
            groupData = self.data.groupby(['ticker', 'timestamp'])
        volsum = groupData['volume'].sum()
        psum = groupData['priceXvolume'].sum()
        vwap = psum / volsum
        o = groupData['price'].first()
        c = groupData['price'].last()
        high = groupData['price'].max()
        low = groupData['price'].min()
        startTime = groupData['timestamp'].first()
        endTime = groupData['timestamp'].last()
        frame = {'Volume': volsum,
                 'VWAP': vwap,
                 'Open': o,
                 'Close': c,
                 'High': high,
                 'Low': low,
                 'numTicks': groupData['price'].count(),
                 'startTime': startTime,
                 'endTime': endTime}


        frame = pd.DataFrame(frame)
        if group == True:
            arrays = [frame.index.get_level_values(0),frame.startTime]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=['ticker','timestamp'])
            frame.index = index
        return frame

    def candlePlot(self, type = 'time', mav = None, start = None, end = None, volume = False):
        plt.figure(figsize=(45,15))
        if type == 'group':
            plotData = self.groupBars.droplevel(axis = 0, level = 0)
        else:
            plotData = self.timeBars.droplevel(axis = 0, level = 0)
        if start and end:
            plotData = plotData.loc[start:end, :]
        if mav:
            mpf.plot(plotData, type = 'candle', mav = mav, volume = volume)
        else:
            mpf.plot(plotData, type='candle', volume = volume)
        plt.show()

    def identifyImbalanceIndexes(self, ET_init, type = 'dollar'):
        ET = ET_init
        ticker = self.data

        if type == 'tick':
            ticks = ticker.ticks
        elif type == 'volume':
            ticks = ticker.ticks * ticker.volume
        else:
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
            barBools = abs_thetas >= thresholds[-1]
            if sum(barBools) == 0:
                T.append(n - i_s[-1])
                i_s.append(n)
                imbalances.append(abs(ticks[i_s[-1]:i].mean()))
                break
            i = barBools[barBools == True].index[0]+1
            imbalances.append(abs(ticks[i_s[-1]:i].mean()))

            T.append(i - i_s[-1])


            abs_thetas[i-1:] = thetas[i-1:] - thetas[i-1]
            abs_thetas[i_s[-1]:i] = 0



            abs_thetas = abs(abs_thetas)
            final_abs_thetas[i:] = abs_thetas[i:]
            i_s.append(i)
            ET = pd.Series(T).ewm(com=0.1).mean().iloc[-1]

            Eimbalance = pd.Series(imbalances).ewm(com=0.1).mean().iloc[-1]
            thresholds.append(ET * Eimbalance)

        return T, i_s, imbalances, final_abs_thetas, thresholds, thetas, ticks



    def identifyRunsIndexes(self, ET_init, type = 'dollar'):
        ET = ET_init
        ticker = self.data

        if type == 'tick':
            ticks = ticker.ticks
        elif type == 'volume':
            ticks = ticker.ticks * ticker.volume
        else:
            ticks = ticker.ticks * ticker.volume * ticker.price


        posticks = ticks.copy(deep=True)
        posticks[posticks < 0] = 0
        negticks = ticks.copy(deep=True)
        negticks[negticks > 0] = 0
        df = pd.DataFrame({"pos": posticks.cumsum(), "neg": -negticks.cumsum()})
        thetas = df[["pos","neg"]].max(axis=1)
        abs_thetas = abs(thetas)
        Eimbalance = max(posticks[0:ET_init][posticks < posticks[0:ET_init].quantile(.99)].mean(),abs(negticks[0:ET_init][negticks > negticks[0:ET_init].quantile(.01)].mean()))
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
            barBools = abs_thetas >= thresholds[-1]
            if sum(barBools) == 0:
                T.append(i - i_s[-1])
                i_s.append(n)
                imbalances.append(max(posticks[i_s[-1]:i].mean(),abs(negticks[i_s[-1]:i].mean())))
                break
            i = barBools[barBools == True].index[1]
            imbalances.append(max(posticks[i_s[-1]:i].mean(),abs(negticks[i_s[-1]:i].mean())))
            T.append(i - i_s[-1])
            if len(imbalances) >= 2 and imbalances[-1] > imbalances[-2]*5:
                imbalances[-1] = imbalances[-2]
                #if T[-1] < T[-2]/3.5:
                T[-1] = T[-2]
            elif len(imbalances) == 1 and imbalances[0] > Eimbalance*5:
                imbalances[0] = Eimbalance
                #if T[0] < ET_init/3.5:
                T[0] = ET_init



            df.pos[i_s[-1]:i-1] = 0
            df.neg[i_s[-1]:i-1] = 0
            df.pos[i-1:] = df.pos[i-1:] - df.pos[i-1]
            df.neg[i-1:] = df.neg[i-1:] - df.neg[i-1]

            abs_thetas = df[["pos","neg"]].max(axis=1)

            final_abs_thetas[i:] = abs_thetas[i:]
            i_s.append(i)

            if sum(T) > ET_init:
                ET = pd.Series(T).ewm(com=0.1).mean().iloc[-1]
                Eimbalance = pd.Series(imbalances).ewm(com=0.1).mean().iloc[-1]



            thresholds.append(ET * Eimbalance)
            #print("T: " + str(T[-1]) + ", imbalance: " + str(imbalances[-1])+ ", Threshold: " + str(thresholds[-1]))

        return T, i_s, imbalances, final_abs_thetas, thresholds, thetas, ticks


    def fixedTimeLabel(self, numBars, type = 'time'):
        if type == 'group':
            returns = self.groupBars.VWAP
        else:
            returns = self.timeBars.VWAP
        print(returns)
        returns = -returns.diff(periods=-numBars)/returns
        tau = returns.std()
        returns[returns > tau] = 1
        returns[returns < -tau] = -1
        returns[(returns > -tau) & (returns < tau)] = 0

        return returns

    def tripleBarrierLabel(self, numBars, upper=3, lower=2, type='time'):
        if type == 'group':
            returns = self.groupBars.VWAP.copy(deep=True)
        else:
            returns = self.timeBars.VWAP.copy(deep=True)

        tau = returns.std()
        returns = pd.DataFrame(returns)
        for i in range(1, numBars+1):
            label = 'shifted: ' + str(i)
            returns[label] = -returns.VWAP.diff(periods=-i)

        shifts = returns.copy(deep=True)
        shifts.VWAP = 0
        shifts = (shifts > upper*tau) | (shifts < -lower*tau)
        shifts[label] = True
        shifts = shifts.cumsum(axis=1)
        shifts = shifts.cumsum(axis=1)
        shifts[shifts != 1] = 0
        returns[shifts != 1] = 0
        label = returns.sum(axis=1)


        print(shifts)
        print(label)





























