
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import  mean_squared_error,confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Thats how the data will look
# in csv file
"""
       Date        Open        High         Low       Close   Adj Close   Volume        
0  2021-06-10  487.170013  490.209991  482.140015  487.269989  487.269989  4382900        
1  2021-06-11  490.000000  491.410004  487.779999  488.769989  488.769989  3124000        
2  2021-06-14  489.679993  503.500000  486.910004  499.890015  499.890015  4400200        
3  2021-06-15  501.230011  501.230011  490.399994  491.899994  491.899994  3104100        
4  2021-06-16  495.000000  496.459991  486.279999  492.410004  492.410004  3533200        
5  2021-06-17  490.250000  501.799988  490.149994  498.339996  498.339996  3198300

# close  price is   current  price
"""
# data path
data='NFLX.csv'
# plt.figure(figsize=(16,8))
# plt.title("Netflix DATA")
# plt.xlabel("Days")
# plt.ylabel("Close price USD($)")
# plt.plot(data['Close'])
# plt.show()

# print(data.shape) #252  trading  days

class stockpredict:
    def __init__(self,data , companyname=None):
        self.data =data
        self.csvdata = pd.read_csv(self.data)
        self.companyname = companyname
    def represent_data_in_gui(self):
        plt.title(f"{self.companyname} Data")
        plt.xlabel("Days")
        plt.ylabel("Close price USD($)")
        plt.plot(self.csvdata['Close'])
        plt.show()
    def predict_data1(self):
        self.csvdata['Date'] = pd.to_datetime(self.csvdata.Date)
        self.csvdata['Close'] =pd.to_numeric(self.csvdata.Close,errors='coerce')
        self.data =self.csvdata.dropna()
        self.xdata = self.csvdata[['Open','High','Low','Volume']]
        self.ydata = self.csvdata['Close']
        self.xtrain,self.xtest,self.ytrain,self.ytest=train_test_split(self.xdata,self.ydata,random_state= 0)
        self.regressor = LinearRegression()
        self.regressor.fit(self.xtrain,self.ytrain)
        print(self.regressor.coef_)
        self.predictedvalue =self.regressor.predict(self.xtest)
        print(self.predictedvalue)


if __name__ == '__main__':
    pr =stockpredict(data)
    pr.predict_data1()