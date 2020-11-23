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

#모델 불러오기
customObjects = {
    'swish' : tf.nn.swish
}
autoencoder_model = load_model(loaded_model, custom_objects= customObjects)
autoencoder_model.summary()


whole_data = np.loadtxt("ppg_bp_filtered.csv",delimiter=',', dtype=np.float32)
bp_data = whole_data[:,:2]
ppg_data = whole_data[:,2:]