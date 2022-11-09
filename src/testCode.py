from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd
import seaborn



# specify 'None' if want to read whole file
# all_stocks_5yr.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('data/all_stocks_5yr.csv', delimiter=',')
df1.dataframeName = 'all_stocks_5yr.csv'



df2 = df1[["date", "close", "Name"]]

df2 = df2.pivot('date','Name', 'close').reset_index()

corr_df = df2.corr(method='pearson')
#reset symbol as index (rather than 0-X)
corr_df.head().reset_index()




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
    plt.show()


def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

#plotCorrelationMatrix(df1, 8)
#plotScatterMatrix(df3, 15, 10)

corrHeatmap(10, 50, corr_df)




