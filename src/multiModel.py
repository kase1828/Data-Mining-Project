import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime

def backtest(data, model, predictors, start=7, step=2):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        # Fit the random forest model
        model.fit(train[predictors], train["Target"])
        
        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0
        
        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
        
        predictions.append(combined)
    
    return pd.concat(predictions)


#@Params:
#string stockName: name of the stock to read data from
#bool indv: If true, it will read from the indv stocks, if false it will read from data
def predictMulti(stockName, indv):
    if(indv):
        df = pd.read_csv('data/indv/' + stockName + '_data.csv')
    else:
        df = pd.read_csv('data/' + stockName)

    df.set_index("date", inplace=True)

    #Test shifting data#
    data = df[["close"]]
    data = data.rename(columns = {'close':'Actual_Close'})

    #print("data:")
    #print(data)

    # Setup our target
    data["Target"] = df.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["close"]
    
    df = df.shift(1)

    predictors = ['open', 'high', 'low', 'close', 'volume']
    data = data.join(df[predictors]).iloc[1:]

    ###
    weekly_mean = data.rolling(7).mean()
    quarterly_mean = data.rolling(90).mean()
    weekly_trend = data.shift(1).rolling(7).mean()["Target"]

    data["weekly_mean"] = weekly_mean["close"] / data["close"]
    data["quarterly_mean"] = quarterly_mean["close"] / data["close"]

    data["weekly_trend"] = weekly_trend

    data["open_close_ratio"] = data["open"] / data["close"]
    data["high_close_ratio"] = data["high"] / data["close"]
    data["low_close_ratio"] = data["low"] / data["close"]


    print("Full data: ")
    print(data)
    
    
    full_predictors = predictors + ["weekly_mean", "quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"]
    ###

    model_predictors = predictors + ["open_close_ratio", "high_close_ratio", "low_close_ratio"]
    

    dataset_train = data.iloc[:-100]
    dataset_test = data.iloc[-100:]

    model = RandomForestClassifier(n_estimators=300, min_samples_split=2, random_state=1)

    
    model.fit(dataset_train[model_predictors], dataset_train["Target"]) ##
    

    

    preds = model.predict(dataset_test[model_predictors]) ##
    preds = pd.Series(preds, index=dataset_test.index)
    precision_score(dataset_test["Target"], preds)

    print(precision_score(dataset_test["Target"], preds))

    
    predictions = backtest(data.iloc[90:], model, full_predictors)
    

    
    print("Prediction percentage after backtesting:")
    print(precision_score(predictions["Target"], predictions["Predictions"]) * 100)

    print(predictions["Predictions"].value_counts())


    

    predictors += ['Actual_Close']

    #predictions = predictions.join(data[predictors])
    print(predictions)
    predictions = predictions.join(dataset_test[predictors])
    
    initialMoney = 1000
    totalMoney = initialMoney
    shares = 0
    trades = 0

    sell = predictions[['Actual_Close','Predictions']]
    sell = sell[-100:]
    #print(dataset_test)
    #print(sell)

    print(sell)



    for i in range(len(sell)):
        if(i % 5 == 0):
            print("Total money: " + str(totalMoney))
        if(sell.iloc[i, 1] == 1.0):
            money = totalMoney // 2
            tradeNow = money // sell.iloc[i,0]
            totalMoney -= (tradeNow * sell.iloc[i,0])
            shares += tradeNow
            trades += 1
            # totalMoney -= sell.iloc[i, 0]
            # shares += 1
            # trades += 1
        if(sell.iloc[i, 1] == 0.0 and shares > 0):
            totalMoney += (shares * sell.iloc[i, 0])
            trades += 1
            shares = 0
            # totalMoney += (sell.iloc[i, 0])
            # shares -= 1
            # trades += 1
    if(shares > 0):
        totalMoney += (shares * sell.iloc[-1,0])
        shares = 0
        trades += 1
    

    startDate = np.datetime64(sell.index[0])
    endDate = np.datetime64(sell.index[-1])

    days = np.busday_count(startDate, endDate)

    

    print("Initial money: " + str(initialMoney))
    print("Total money: " + str(totalMoney))
    print("Total shares: " + str(shares))
    print("Total trades: " + str(trades))
    print("Percent gain or loss: " + str((totalMoney/initialMoney - 1) * 100) + '%')
    print("Trading days: " + str(days))

    return(totalMoney - initialMoney)


    


predictMulti('AMD', True)

