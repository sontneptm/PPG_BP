from matplotlib import pyplot as plt
from keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import Adam

#불러올 모델명
loaded_model = 'ppg_gru_model.h5'
#bp_data = np.loadtxt('ppg_bp_encoded.csv', delimiter=',') #데이터 읽어옴
bp_data = np.loadtxt('ppg_jeong_encoded.csv', delimiter=',') #데이터 읽어옴

customObjects = {
    'swish' : tf.nn.swish
}

y_data = bp_data[:, :2]
x_data = bp_data[:, 2:]

print('x shape : ', x_data.shape)  # (4,3)
print('y shape : ', y_data.shape)  # (4,)

print('x_data shape : ', x_data.shape)
print('-------x reshape-----------')
x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))

model = load_model(loaded_model, custom_objects= customObjects)
model.compile(optimizer=Adam(lr=1.46e-3), loss='mse')
# 3. 모델 사용하기
yhat = model.predict(x_data) #예측 결과
print(yhat)
for i in yhat :
    print(i)

sys_list = []
dia_list = []
for i in range(len(y_data)) :
    sys_list.append(yhat[i][0]-y_data[i][0])
    dia_list.append(yhat[i][1]-y_data[i][1])

plt.rcParams["figure.figsize"] = (7,5)
plt.subplot(211)
plt.title('systolic')
plt.hist(sys_list, bins = 100, histtype='step')

plt.subplot(212)
plt.title('diastolic')
plt.hist(dia_list, bins = 100,histtype='stepfilled')
plt.show()