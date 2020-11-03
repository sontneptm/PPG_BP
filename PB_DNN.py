import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

xy_data = np.loadtxt("ppg_bp.csv",delimiter=',', dtype=np.float32)

xd = xy_data[:,2:]
yd = xy_data[:,:2]

xd = (xd-xd.min())/(xd.max()-xd.min())
yd = yd

#hyper params
SPLIT_RATE = 0.2
EPOCH = 200
INPUT_SIZE = len(xd[0])

xt, xv, yt, yv = train_test_split(xd, yd, test_size = SPLIT_RATE, random_state = 123)
xt = np.array(xt)
xv = np.array(xv)
yt = np.array(yt)
yv = np.array(yv)

with tf.device('GPU:0'):
    tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=0)
    model = models.Sequential()
    model.add(layers.Dense(4096, activation='linear', input_shape=[INPUT_SIZE]))
    model.add(layers.Dense(2048, activation='linear'))
    model.add(layers.Dense(1024, activation='linear'))
    model.add(layers.Dense(512, activation='linear'))
    model.add(layers.Dense(256, activation='linear'))
    model.add(layers.Dense(2, activation='linear'))

    model.compile(optimizer=Adam(lr=1.46e-3), loss='mse')
    hist = model.fit(xt, yt, validation_split=0.2, shuffle=True, epochs=EPOCH)
    print(hist)
    model.summary()

    pd = model.predict(xv)

    for i in range(len(xv)):
        #fmt = '실제값: {1}, 예측값: {2:.5f} {3:.5f}, 정제된값: {4:.0f} {5:.0f}'
        #print(fmt.format(yv[i], pd[i][0], pd[i][1], pd[i][0], pd[i][1]))
        print("실제값 : ", yv[i],"\t", "예측값 :", pd[i])
    
