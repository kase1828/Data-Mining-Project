import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

df = pd.read_csv('data/indv/AMD_data.csv')
name = df.iloc[0, df.columns.get_loc('Name')]

def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y 

df = df['open'].values
df = df.reshape(-1, 1)

print(df.shape)

print(df)
dataset_train = np.array(df[:int(df.shape[0]*0.8)])#Train set
dataset_test = np.array(df[int(df.shape[0]*0.8):])#Test set
print(dataset_train.shape)
print(dataset_test.shape)

scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_train[:5]

dataset_test = scaler.transform(dataset_test)
dataset_test[:5]

x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)


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

#Turn data into 3D array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model.compile(loss='mean_squared_error', optimizer='adam')

#Train model#
#model.fit(x_train, y_train, epochs=100, batch_size=40)
#model.save('stock_prediction.h5') #Save prediction model to this name in the root directory
#Train model#

model = load_model('IBM_by_close.h5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(16,8))
ax.set_facecolor('#000041')
ax.plot(y_test_scaled, color='red', label='Actual price')
plt.plot(predictions, color='cyan', label='Predicted price')
plt.legend()
plt.title("Predicting: " + name)

plt.show()
