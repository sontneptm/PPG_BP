import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models

whole_data = np.loadtxt("ppg_bp_filtered.csv",delimiter=',', dtype=np.float32)
bp_data = whole_data[:,:2]
ppg_data = whole_data[:,2:]
#ppg_data = (ppg_data-ppg_data.min())/(ppg_data.max()-ppg_data.min())

LATENT_DIM = 32
SPLIT_RATE = 0.2
EPOCHS = 10
PPG_LENGTH = len(ppg_data[0])
checkpoint_path = "ppg_bp_aae/pba.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

pt, pv = train_test_split(ppg_data, test_size = SPLIT_RATE, random_state = 123)

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            layers.Dense(1024, activation=tf.nn.swish),
            layers.Dense(512, activation=tf.nn.swish),
            layers.Dense(latent_dim),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(512, activation=tf.nn.swish),
            layers.Dense(1024, activation=tf.nn.swish),
            layers.Dense(PPG_LENGTH, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(LATENT_DIM)

autoencoder.compile(optimizer=Adam(lr=1.46e-4), loss=losses.MeanSquaredError())
autoencoder.fit(pt, pt, epochs=EPOCHS, shuffle=True, validation_data=(pv, pv))
autoencoder.summary()

encoded_val_data = autoencoder.encoder(pv).numpy()
decoded_val_data = autoencoder.decoder(encoded_val_data).numpy()

x_axis = range(len(ppg_data[0]))
plt.plot(x_axis, decoded_val_data[5])
plt.plot(x_axis, pv[5])
plt.show()

file = open('ppg_jeong_encoded.csv','a')

realtime_data = np.loadtxt("ppg_jeong_filtered.csv",delimiter=',', dtype=np.float32)
realtime_bp_data = whole_data[:,:2]
realtime_ppg_data = whole_data[:,2:]



encoded_data = autoencoder.encoder(realtime_ppg_data).numpy()
print(len(encoded_data))
for i in range(len(encoded_data)):
    tmp_list = []
    tmp_list.append(bp_data[i].tolist()[0])
    tmp_list.append(bp_data[i].tolist()[1])

    for d in encoded_data[i] :
        tmp_list.append(d)

    rtn = str(tmp_list)[1:-1]
    file.write(str.format(rtn) + "\n")
