import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

whole_data = np.loadtxt("ppg_bp_filtered.csv",delimiter=',', dtype=np.float32)
bp_data = whole_data[:,:2]
ppg_data = whole_data[:,2:]
ppg_data = (ppg_data-ppg_data.min())/(ppg_data.max()-ppg_data.min())

LATENT_DIM = 64
SPLIT_RATE = 0.2
EPOCHS = 10000
PPG_LENGTH = len(ppg_data[0])

pt, pv = train_test_split(ppg_data, test_size = SPLIT_RATE, random_state = 123)

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            layers.Dense(2048, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(2048, activation='relu'),
            layers.Dense(PPG_LENGTH, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(LATENT_DIM)
autoencoder.compile(optimizer=Adam(lr=1.46e-3), loss=losses.MeanSquaredError())
autoencoder.fit(pt, pt, epochs=EPOCHS, shuffle=True, validation_data=(pv, pv))

encoded_imgs = autoencoder.encoder(pv).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

x_axis = range(len(ppg_data[0]))
plt.plot(x_axis, decoded_imgs[5])
plt.plot(x_axis, pv[5])
plt.show()