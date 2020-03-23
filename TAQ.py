import numpy as np
import pandas as pd
import sklearn
import datetime

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
        frame = {'volume': volsum, 'vwap': vwap, 'open': o, 'close': c, 'high': high, 'low': low}
        frame = pd.DataFrame(frame)
        return frame













