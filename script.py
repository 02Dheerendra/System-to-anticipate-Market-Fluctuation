from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import os
import numpy as np
import math
import datetime as dt
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import max_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Bidirectional, TimeDistributed
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.metrics import RootMeanSquaredError

from itertools import cycle
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

app = Flask(__name__)

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        Source = request.form["Source"]
        if (Source == 'asianpaint'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/ASIANPAINT.BO.csv')

        elif (Source == 'axisbank'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/AXISBANK.BO (1).csv')

        elif (Source == 'bajajauto'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/BAJAJ-AUTO.BO.csv')

        elif (Source == 'bhartiartl'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/BHARTIARTL.BO.csv')
            
        elif (Source == 'cipla'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/CIPLA.BO.csv')

        elif (Source == 'hcltech'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/HCLTECH.BO.csv')
            
        elif (Source == 'hindunilvr'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/HINDUNILVR.NS.csv')
            
        elif (Source == 'icicibank'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/ICICIBANK.BO.csv')
            
        elif (Source == 'infy'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/INFY.BO.csv')
            
        elif (Source == 'itc'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/ITC.BO.csv')
            
        elif (Source == 'kotakbank'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/KOTAKBANK.BO.csv')
            
        elif (Source == 'reliance'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/RELIANCE.NS.csv')
            
        elif (Source == 'tatamotors'):
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/TATAMOTORS.BO.csv')
            
        else:
            stock_df = pd.read_csv('C:/Users/Dell/Downloads/dataset for final year project/TCS.BO (1).csv')


        Type = request.form["Destination"]
        if (Type == 'LSTM'):
        
            stock_df = stock_df.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                            'Adj Close':'adj_close','Volume':'volume'})
            stock_df.dropna(inplace=True)
            stock_df.info()

            stock_df['date'] = pd.to_datetime(stock_df['date'], utc=True)

            OHLC_avg = stock_df[['open','high', 'low', 'close']].mean(axis=1)
            HLC_avg = stock_df[['high', 'low', 'close']].mean(axis = 1)

            stock_df_OHLC=pd.DataFrame({'date':stock_df['date'],'close':OHLC_avg})

            stock_df_OHLC_Orignal=stock_df_OHLC.copy()

            training_size=int(len(stock_df_OHLC)*0.80)

            test_size=len(stock_df_OHLC)-training_size

            train_data,test_data = stock_df_OHLC[0:training_size], stock_df_OHLC[training_size:len(stock_df_OHLC)]

            del train_data['date']

            del test_data['date']

            scaler=MinMaxScaler(feature_range=(0,1))

            train_data=scaler.fit_transform(np.array(train_data).reshape(-1,1))

            test_data = scaler.transform(np.array(test_data).reshape(-1,1))

            def create_sliding_window(dataset, time_step=1):
                dataX, dataY = [], []
                for i in range(len(dataset)-time_step-1):
                    a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----45   46
                    dataX.append(a)
                    dataY.append(dataset[i + time_step, 0])
                return np.array(dataX), np.array(dataY)

            time_step = 45

            X_train, y_train = create_sliding_window(train_data, time_step)

            X_test, y_test = create_sliding_window(test_data, time_step)

            X_train =X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)

            X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

            model=Sequential()
            model.add(LSTM(50,return_sequences=True,input_shape=(45,1)))
            model.add(Dropout(rate=0.2))
            model.add(LSTM(50,return_sequences=True))
            model.add(Dropout(rate=0.3))
            model.add(LSTM(50))
            model.add(Dropout(rate=0.2))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error',optimizer='adam')

            model.summary()

            history = model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=20, batch_size=64, verbose=1)

            del stock_df_OHLC['date']

            look_back=time_step

            x_input=test_data[len(test_data)-time_step:].reshape(1,-1) # time_step is 45
            temp_input=list(x_input)

            outx_input = x_input.reshape((1, 45,1))
            output=model.predict(outx_input, verbose=1)
            output=scaler.inverse_transform(output)
            temp_input=temp_input[0].tolist()

            from numpy import array

            lst_output=[]
            n_steps=time_step
            i=0
            pred_days = 30
            while(i<pred_days):

                if(len(temp_input)>time_step):

                    x_input=np.array(temp_input[1:])
                    #print("{} day input {}".format(i,x_input))
                    x_input = x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))

                    y_pred = model.predict(x_input, verbose=1)
                    #print("{} day output {}".format(i,y_pred))
                    temp_input.extend(y_pred[0].tolist())
                    temp_input=temp_input[1:]
                    #print(temp_input)

                    lst_output.extend(y_pred.tolist())
                    i=i+1

                else:

                    x_input = x_input.reshape((1, n_steps,1)) # Reshape x_input to a 3D Tensor [samples, time steps, features] before feeding into the model
                    y_pred = model.predict(x_input, verbose=0)
                    temp_input.extend(y_pred[0].tolist())

                    lst_output.extend(y_pred.tolist())
                    i=i+1

            last_days=np.arange(1,time_step+1)
            day_pred=np.arange(time_step+1,time_step+pred_days+1)

            temp_matrix = np.empty((len(last_days)+pred_days+1, 1))

            temp_matrix[:] = np.nan

            temp_matrix.shape

            temp_matrix = temp_matrix.reshape(1,-1).tolist()[0]


            last_original_days_value = temp_matrix

            next_predicted_days_value = temp_matrix

            last_original_days_value[0:time_step+1] = stock_df_OHLC_Orignal[len(stock_df_OHLC_Orignal)-time_step:]['close'].tolist()

            next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]


            last_original_days_value[0:time_step+1]


            next_predicted_days_value[time_step+1:]

            new_pred_plot = pd.DataFrame({
                'last_original_days_value':last_original_days_value,
                'next_predicted_days_value':next_predicted_days_value
            })
            names = cycle(['Last 45 days close price','Predicted next 30 days close price'])

            fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'], new_pred_plot['next_predicted_days_value']],
                          labels={'value': 'Stock price','index': 'Timestamp'})

            fig.update_layout(title_text='Compare last 45 days vs next 30 days',
                              plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

            fig.for_each_trace(lambda t:  t.update(visible=True, name = next(names)))

            fig.update_xaxes(showgrid=False)

            fig.update_yaxes(showgrid=False)

            graph1JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            stock_df_c = scaler.transform(np.array(stock_df_OHLC).reshape(-1,1))
            lstmdf=stock_df_c.tolist()

            lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())

            lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

            names = cycle(['Close price'])

            fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})

            fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                              plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

            fig.for_each_trace(lambda t:  t.update(name = next(names)))

            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            graph2JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            
        elif(Type == 'CNN'):
            stock_df = stock_df.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                            'Adj Close':'adj_close','Volume':'volume'})
            stock_df.dropna(inplace=True)

            stock_df['date'] = pd.to_datetime(stock_df['date'], utc=True)

            data=stock_df

            OHLC_avg = stock_df[['open','high', 'low', 'close']].mean(axis=1)
            HLC_avg = stock_df[['high', 'low', 'close']].mean(axis = 1)

            stock_df_OHLC=pd.DataFrame({'date':stock_df['date'],'close':OHLC_avg})

            stock_df_OHLC_Orignal=stock_df_OHLC.copy()

            training_size=int(len(stock_df_OHLC)*0.80)

            test_size=len(stock_df_OHLC)-training_size

            train_data,test_data = stock_df_OHLC[0:training_size], stock_df_OHLC[training_size:len(stock_df_OHLC)]

            del train_data['date']

            del test_data['date']

            scaler=MinMaxScaler(feature_range=(0,1))

            train_data=scaler.fit_transform(np.array(train_data).reshape(-1,1))

            test_data = scaler.transform(np.array(test_data).reshape(-1,1))

            def create_sliding_window(dataset, time_step=1):
                dataX, dataY = [], []
                for i in range(len(dataset)-time_step-1):
                    a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
                    dataX.append(a)
                    dataY.append(dataset[i + time_step, 0])
                return np.array(dataX), np.array(dataY)

            time_step = 45

            X_train, y_train = create_sliding_window(train_data, time_step)

            X_test, y_test = create_sliding_window(test_data, time_step)

            X_train =X_train.reshape(X_train.shape[0], 1,X_train.shape[1] ,1)

            X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1] , 1)

            X_train

            model=Sequential()
            model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=(None,45, 1))))
            model.add(TimeDistributed(MaxPooling1D(2)))
            model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
            model.add(TimeDistributed(MaxPooling1D(2)))
            model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
            model.add(TimeDistributed(MaxPooling1D(2)))
            model.add(TimeDistributed(Flatten()))

            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(64, return_sequences=False)))
            model.add(Dropout(0.2))

            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse',metrics=['mse', 'mae'])
            history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20,batch_size=40, verbose=1, shuffle =True)

            model.summary()

            train_predict=model.predict(X_train)

            test_predict=model.predict(X_test)

            train_predict.shape, test_predict.shape

            train_predict = scaler.inverse_transform(train_predict)

            test_predict = scaler.inverse_transform(test_predict)

            original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))

            original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

            del stock_df_OHLC['date']

            look_back=time_step

            train_predict_look_ahead = np.empty_like(stock_df_OHLC)

            train_predict_look_ahead[:, :] = np.nan

            train_predict_look_ahead[look_back:len(train_predict)+look_back, :] = train_predict
            test_predict_look_ahead = np.empty_like(stock_df_OHLC)

            test_predict_look_ahead[:, :] = np.nan

            test_predict_look_ahead[len(train_predict)+(look_back*2)+1:len(stock_df_OHLC)-1, :] = test_predict

            x_input=test_data[len(test_data)-time_step:].reshape(1,-1) # time_step is 15

            temp_input=list(x_input)

            outx_input = x_input.reshape((1, 1,45,1))
            output=model.predict(outx_input, verbose=0)

            output=scaler.inverse_transform(output)[0]

            temp_input=temp_input[0].tolist()

            from numpy import array

            lst_output=[]
            n_steps=time_step
            i=0
            pred_days = 30
            while(i<pred_days):

                if(len(temp_input)>time_step):

                    x_input=np.array(temp_input[1:])
                    #print("{} day input {}".format(i,x_input))
                    x_input = x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, 1,n_steps, 1))

                    y_pred = model.predict(x_input, verbose=0)
                    #print("{} day output {}".format(i,y_pred))
                    temp_input.extend(y_pred[0].tolist())
                    temp_input=temp_input[1:]
                    #print(temp_input)

                    lst_output.extend(y_pred.tolist())
                    i=i+1

                else:

                    x_input = x_input.reshape((1,1, n_steps,1)) # Reshape x_input to a 3D Tensor [samples, time steps, features] before feeding into the model
                    y_pred = model.predict(x_input, verbose=0)
                    temp_input.extend(y_pred[0].tolist())

                    lst_output.extend(y_pred.tolist())
                    i=i+1

            last_days=np.arange(1,time_step+1)
            day_pred=np.arange(time_step+1,time_step+pred_days+1)


            temp_matrix = np.empty((len(last_days)+pred_days+1, 1))

            temp_matrix[:] = np.nan

            temp_matrix.shape

            temp_matrix = temp_matrix.reshape(1,-1).tolist()[0]


            last_original_days_value = temp_matrix

            next_predicted_days_value = temp_matrix

            last_original_days_value[0:time_step+1] = stock_df_OHLC_Orignal[len(stock_df_OHLC_Orignal)-time_step:]['close'].tolist()

            next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]


            last_original_days_value[0:time_step+1]


            next_predicted_days_value[time_step+1:]

            new_pred_plot = pd.DataFrame({
                'last_original_days_value':last_original_days_value,
                'next_predicted_days_value':next_predicted_days_value
            })

            names = cycle(['Last 150 days close price','Predicted next 30 days close price'])

            fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'], new_pred_plot['next_predicted_days_value']],
                          labels={'value': 'Stock price','index': 'Timestamp'})

            fig.update_layout(title_text='Compare last 150 days vs next 30 days',
                              plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

            fig.for_each_trace(lambda t:  t.update(visible=True, name = next(names)))

            fig.update_xaxes(showgrid=False)

            fig.update_yaxes(showgrid=False)

            graph1JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


            stock_df_c = scaler.transform(np.array(stock_df_OHLC).reshape(-1,1))
            lstmdf=stock_df_c.tolist()

            lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())

            lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

            names = cycle(['Close price'])

            fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})

            fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                              plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

            fig.for_each_trace(lambda t:  t.update(name = next(names)))

            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            graph2JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


        else:
        
            stock_df = stock_df.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                            'Adj Close':'adj_close','Volume':'volume'})
            stock_df.dropna(inplace=True)
            stock_df.info()

            stock_df['date'] = pd.to_datetime(stock_df['date'], utc=True)

            OHLC_avg = stock_df[['open','high', 'low', 'close']].mean(axis=1)
            HLC_avg = stock_df[['high', 'low', 'close']].mean(axis = 1)

            stock_df_OHLC=pd.DataFrame({'date':stock_df['date'],'close':OHLC_avg})

            stock_df_OHLC_Orignal=stock_df_OHLC.copy()

            training_size=int(len(stock_df_OHLC)*0.80)

            test_size=len(stock_df_OHLC)-training_size

            train_data,test_data = stock_df_OHLC[0:training_size], stock_df_OHLC[training_size:len(stock_df_OHLC)]

            del train_data['date']

            del test_data['date']

            scaler=MinMaxScaler(feature_range=(0,1))

            train_data=scaler.fit_transform(np.array(train_data).reshape(-1,1))

            test_data = scaler.transform(np.array(test_data).reshape(-1,1))

            def create_sliding_window(dataset, time_step=1):
                dataX, dataY = [], []
                for i in range(len(dataset)-time_step-1):
                    a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----45   46
                    dataX.append(a)
                    dataY.append(dataset[i + time_step, 0])
                return np.array(dataX), np.array(dataY)

            time_step = 45

            X_train, y_train = create_sliding_window(train_data, time_step)

            X_test, y_test = create_sliding_window(test_data, time_step)

            X_train =X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)

            X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
            model=Sequential()
            model.add(GRU(50,return_sequences=True,input_shape=(time_step,1)))
            model.add(GRU(50,return_sequences=True))
            model.add(GRU(50))
            model.add(Dropout(rate=0.4))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error',optimizer='adam')

            model.summary()

            history = model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=20, batch_size=64, verbose=1)

            del stock_df_OHLC['date']

            look_back=time_step

            x_input=test_data[len(test_data)-time_step:].reshape(1,-1) # time_step is 45
            temp_input=list(x_input)

            outx_input = x_input.reshape((1, 45,1))
            output=model.predict(outx_input, verbose=1)
            output=scaler.inverse_transform(output)
            temp_input=temp_input[0].tolist()

            from numpy import array

            lst_output=[]
            n_steps=time_step
            i=0
            pred_days = 30
            while(i<pred_days):

                if(len(temp_input)>time_step):

                    x_input=np.array(temp_input[1:])
                    #print("{} day input {}".format(i,x_input))
                    x_input = x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))

                    y_pred = model.predict(x_input, verbose=1)
                    #print("{} day output {}".format(i,y_pred))
                    temp_input.extend(y_pred[0].tolist())
                    temp_input=temp_input[1:]
                    #print(temp_input)

                    lst_output.extend(y_pred.tolist())
                    i=i+1

                else:

                    x_input = x_input.reshape((1, n_steps,1)) # Reshape x_input to a 3D Tensor [samples, time steps, features] before feeding into the model
                    y_pred = model.predict(x_input, verbose=0)
                    temp_input.extend(y_pred[0].tolist())

                    lst_output.extend(y_pred.tolist())
                    i=i+1

            last_days=np.arange(1,time_step+1)
            day_pred=np.arange(time_step+1,time_step+pred_days+1)

            temp_matrix = np.empty((len(last_days)+pred_days+1, 1))

            temp_matrix[:] = np.nan

            temp_matrix.shape

            temp_matrix = temp_matrix.reshape(1,-1).tolist()[0]


            last_original_days_value = temp_matrix

            next_predicted_days_value = temp_matrix

            last_original_days_value[0:time_step+1] = stock_df_OHLC_Orignal[len(stock_df_OHLC_Orignal)-time_step:]['close'].tolist()

            next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]


            last_original_days_value[0:time_step+1]


            next_predicted_days_value[time_step+1:]

            new_pred_plot = pd.DataFrame({
                'last_original_days_value':last_original_days_value,
                'next_predicted_days_value':next_predicted_days_value
            })
            names = cycle(['Last 45 days close price','Predicted next 30 days close price'])

            fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'], new_pred_plot['next_predicted_days_value']],
                          labels={'value': 'Stock price','index': 'Timestamp'})

            fig.update_layout(title_text='Compare last 45 days vs next 30 days',
                              plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')

            fig.for_each_trace(lambda t:  t.update(visible=True, name = next(names)))

            fig.update_xaxes(showgrid=False)

            fig.update_yaxes(showgrid=False)

            graph1JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            stock_df_c = scaler.transform(np.array(stock_df_OHLC).reshape(-1,1))
            lstmdf=stock_df_c.tolist()

            lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())

            lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

            names = cycle(['Close price'])

            fig = px.line(lstmdf,labels={'value': 'Stock price','index': 'Timestamp'})

            fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                              plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Stock')

            fig.for_each_trace(lambda t:  t.update(name = next(names)))

            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            graph2JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            

            

        return render_template('home.html',prediction_text="Price of next day is Rs. {}".format(output),graph1JSON=graph1JSON,  graph2JSON=graph2JSON)


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)
