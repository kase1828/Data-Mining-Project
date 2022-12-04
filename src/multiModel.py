import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

def backtest(data, model, predictors, start=10, step=15):
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

def predictMulti(stockName):
    df = pd.read_csv('data/indv/' + stockName + '_data.csv')

    df.set_index("date", inplace=True)

    #Test shifting data#
    data = df[["close"]]
    data = data.rename(columns = {'close':'Actual_Close'})

    print("data:")
    print(data)

    # Setup our target
    data["Target"] = df.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["close"]
    
    df = df.shift(1)

    predictors = ['open', 'high', 'low', 'close', 'volume']
    data = data.join(df[predictors]).iloc[1:]
    print("data:")
    print(data)

    dataset_train = data.iloc[:-100]
    dataset_test = data.iloc[-100:]

    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

    model.fit(dataset_train[predictors], dataset_train["Target"])

    preds = model.predict(dataset_test[predictors])
    preds = pd.Series(preds, index=dataset_test.index)
    precision_score(dataset_test["Target"], preds)

    print(precision_score(dataset_test["Target"], preds))

    comparison = pd.concat({"Target": dataset_test["Target"],"Predictions": preds}, axis=1)
    comparison.plot()
    #plt.show()
    
    predictions = backtest(data, model, predictors)

    predictions["Predictions"].value_counts()
    print(predictions)

    print(predictions["Predictions"].value_counts())
    print(predictions["Target"].value_counts())
    
    print(precision_score(predictions["Target"], predictions["Predictions"]))

    weekly_mean = data.rolling(7).mean()
    quarterly_mean = data.rolling(90).mean()
    annual_mean = data.rolling(365).mean()
    weekly_trend = data.shift(1).rolling(7).mean()["Target"]

    data["weekly_mean"] = weekly_mean["close"] / data["close"]
    data["quarterly_mean"] = quarterly_mean["close"] / data["close"]
    data["annual_mean"] = annual_mean["close"] / data["close"]

    data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
    data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
    data["weekly_trend"] = weekly_trend

    data["open_close_ratio"] = data["open"] / data["close"]
    data["high_close_ratio"] = data["high"] / data["close"]
    data["low_close_ratio"] = data["low"] / data["close"]

    full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"]
    predictions = backtest(data.iloc[365:], model, full_predictors)

    print(precision_score(predictions["Target"], predictions["Predictions"]))

    print(predictions["Predictions"].value_counts())



    

predictMulti('AMD')