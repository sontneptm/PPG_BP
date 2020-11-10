import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

xy_data = np.loadtxt("ppg_bp_encoded.csv",delimiter=',', dtype=np.float32)
xd = xy_data[:,2:]
yd = xy_data[:,:2]

xd = xd.reshape(-1,32,1)
#xd = (xd-xd.min())/(xd.max()-xd.min())

EPOCH = 1000
SPLIT_RATE = 0.2

xt, xv, yt, yv = train_test_split(xd, yd, test_size = SPLIT_RATE, random_state = 123)

model = models.Sequential()
model.add(Conv1D(filters=128, kernel_size=2, activation=tf.nn.swish, input_shape=(32,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=2, activation=tf.nn.swish))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=4, activation=tf.nn.swish))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(500, activation=tf.nn.swish))
model.add(Dense(500, activation=tf.nn.swish))
model.add(Dense(500, activation=tf.nn.swish))
model.add(Dense(500, activation=tf.nn.swish))
model.add(Dense(2, activation=tf.nn.swish))
model.compile(optimizer = 'adam', loss='mse')
model.fit(xt, yt, epochs=EPOCH, verbose=2, validation_split=0.2)
model.summary()

pd = model.predict(xv)

for i in range(len(xv)):
    #fmt = '실제값: {1}, 예측값: {2:.5f} {3:.5f}, 정제된값: {4:.0f} {5:.0f}'
    #print(fmt.format(yv[i], pd[i][0], pd[i][1], pd[i][0], pd[i][1]))
    print("실제값 : ", yv[i],"\t", "예측값 :", pd[i])

print(r2_score(yv, pd))
