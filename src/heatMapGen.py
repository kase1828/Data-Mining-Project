from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd
import seaborn




dfAll = pd.read_csv('data/all_stocks_5yr.csv', delimiter=',')
dfAll.dataframeName = 'all_stocks_5yr.csv'


#Get a correlation table between all stocks
#Parameters:
#   dataFrame: An unprocessed dataframe that contains stock data from /data
#   value: Which metric to consider, ie. volume, close, high, etc.
def getCorrTable(dataFrame, value):
    corrData = dataFrame[["date", value, "Name"]]
    corrData = corrData.pivot('date', 'Name', f'{value}')
    corrData = corrData.corr(method='pearson')
    corrData.head().reset_index()
    corrData.columns.name = value

    return corrData

#Generates a heatmap utilizing the correlation dataframe.
#Paramters: 
#   numOfStock: How many stocks to correlate in heatmap
#   startPoint: Index to start from
#   corr_df: The correlized dataframe of stocks that has been pivoted.
def corrHeatmap(numOfStocks, startPoint, corr_df):
    corr_df = corr_df.iloc[startPoint:startPoint+numOfStocks, startPoint:startPoint+numOfStocks]
    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask)] = True
    #generate plot
    seaborn.heatmap(corr_df, cmap='RdYlGn', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90) 
    plt.title(corr_df.columns.name)
    plt.show()

testCorr = getCorrTable(dfAll, "high")


corrHeatmap(15, 0, testCorr, )

print(testCorr.columns.name)

#print(testCorr)