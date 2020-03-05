import numpy as np
import pandas as pd
import sklearn
import datetime

class TAQ():
    """TAQ object generated from WRDS database TAQ trade csv file.
    Stores initial CSV as pd dataframe and renames columns"""

    def __init__(self, path):
        self.taqPath = path
        self.taqData = pd.read_csv(path)


    def



