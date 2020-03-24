from TAQ import TAQ
import pandas as pd

def main():
    d = pd.read_csv('QQQ1day.csv')
    taq = TAQ(data=d)
    print(taq.data.head(-10))





if __name__ == '__main__':
    main()