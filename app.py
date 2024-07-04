import yfinance as yf

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st


st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker','IDFCFIRSTB.NS')

idfc = yf.Ticker(user_input)

idfc = idfc.history(period="max")

idfc.drop(columns=['Dividends','Stock Splits'],inplace = True)

idfc.index = idfc.index.strftime('%Y-%m-%d')

df = idfc 

#Describing data
st.subheader('Complete Data')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price v/s Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close,'g')
st.pyplot(fig)

st.subheader('Closing Price v/s Time chart with 100MA & 50MA')
ma100 = df.Close.rolling(100).mean()
ma50 = df.Close.rolling(50).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma50,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

# Splitting data into training and testing
#taking 70:30 ratio to split data
X = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
y = pd.DataFrame(df['Close'][int(len(df)*0.70):])

#scaling the data
from sklearn.preprocessing import * 
M = MinMaxScaler(feature_range=(0,1))

X_array = M.fit_transform(X)


#loading the built model
model = load_model('my_model.keras')


#predict values for testing data, by taking data from training set 
past_100_days = X.tail(100)

final_df = pd.concat([past_100_days,y],ignore_index=True)
input_data = M.fit_transform(final_df)

xtest=[]
ytest=[]

for i in range(100,input_data.shape[0]):
    xtest.append(input_data[i-100:i])
    ytest.append(input_data[i,0])

xtest,ytest = np.array(xtest),np.array(ytest)
ypred = model.predict(xtest)

scaler = M.scale_
scalefactor = 1/scaler[0]

ypred = ypred*scalefactor
ytest = ytest*scalefactor

#Final graph
st.subheader('Prediction v/s Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(ytest,'b',label = 'Original Price')
plt.plot(ypred,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
