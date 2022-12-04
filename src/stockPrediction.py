import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout




def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y 

def getModelTrain(stockName, attribute, epochs_, batchSize_):
    #df = pd.read_csv('data/indv/' + stockName + '_data.csv')
    df = pd.read_csv('data/sp_20yr.csv')

    df.set_index("date", inplace=True)

    #Test shifting data#
    data = df[["close"]]
    data = data.rename(columns = {'close':'Actual_Close'})

    print("data:")
    print(data)

    # Setup our target
    data["Target"] = df.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["close"]
    print("data after target:")
    print(data)
    data = data.iloc[:-100]
    print("data after shift")
    print(data)

    print("df:")
    print(df)
    df = df.shift(1)
    print("df after shift:")
    print(df)

    df = df.iloc[:-100]##
    print("df after iloc:")
    print(df)

    predictors = ['open', 'high', 'low', 'close', 'volume']
    data = data.join(df[predictors]).iloc[1:]
    print("data:")
    print(data)

    df = data

    #Test shifting data#

    df = df[attribute].values
    df = df.reshape(-1, 1)
    print("df:")
    print(df)
    print(df.shape)
    print("Dataset train:")
    #dataset_train = np.array(df[:int(df.shape[0]*0.8)])#Train set
    dataset_train = np.array(df)
    print(dataset_train.shape)

    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_train[:5]

    print(dataset_train.shape)

    

    

    x_train, y_train = create_dataset(dataset_train)

    #Training model parameters#
    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    #Training model parameters#

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    

    model.compile(loss='mean_squared_error', optimizer='adam')

    #Train model#
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batchSize_)
    model.save(stockName+'_by_'+attribute+'.h5') #Save prediction model to this name in the root directory
    #Train model#

def predictPrice(stockName, attribute, modelName):
    df = pd.read_csv('data/indv/' + stockName + '_data.csv')
    df.set_index("date", inplace=True)
    print(df)

    df_train = df
    df_train = df_train.iloc[:-100]
    df_train = df_train[attribute].values
    df_train = df_train.reshape(-1, 1)

    print('Last 100 rows for testing: ')
    df = df.iloc[-100:]##
    print(df)
    print("Get only the attribute column: ")
    df = df[attribute].values
    print(df)
    df = df.reshape(-1, 1)

    print(df)
    
    print('Dataset_test: ')
    #dataset_test = np.array(df[int(df.shape[0]*0.8):])#Test set
    dataset_test = np.array(df)
    print(dataset_test)
    print(dataset_test.shape)
    
    #Need "Training" dataset in order to properly fit a scalar 
    print("dataset train:")
    #dataset_train = np.array(df[:int(df_train.shape[0]*0.8)])
    dataset_train = np.array(df_train)
    print(dataset_train)
    print(dataset_train.shape)
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_train = scaler.fit_transform(dataset_train)
    dataset_train[:5]
    

    dataset_test = scaler.transform(dataset_test)
    dataset_test[:5]
    
    x_test, y_test = create_dataset(dataset_test)

    #Turn data into 3D array
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    model = load_model(modelName+'.h5')

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('#000041')
    ax.plot(y_test_scaled, color='red', label='Actual price')
    plt.plot(predictions, color='cyan', label='Predicted price')
    plt.legend()
    plt.title("Predicting: " + stockName)

    plt.show()






#getModelTrain('sp_20yr','close', 50, 32)

predictPrice('AAPL', 'close', 'sp_20yr_by_close')



