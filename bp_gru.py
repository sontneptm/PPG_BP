from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.layers import Dense, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from tensorflow.keras.optimizers import Adam

# Open, High, Low, Volume, Close
bp_data = np.loadtxt('ppg_bp_encoded.csv', delimiter=',') #데이터 읽어옴

y_data = bp_data[:, :2]
x_data = bp_data[:, 2:]

y_test, y_train, x_test, x_train = train_test_split(y_data, x_data, test_size=0.8, shuffle= True)

#print(x_data)
print('-------x reshape-----------')
x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))  # (4,3,1) reshape 전체 곱 수 같아야 4*3=4*3*1
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))  # (4,3,1) reshape 전체 곱 수 같아야 4*3=4*3*1


# 2. 모델 구성
model = Sequential()
model.add(GRU(100, activation='swish', input_shape=(x_data.shape[1], 1)))
# DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
#model.add(GRU(100))
model.add(Dense(128, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dense(512, activation='swish'))
model.add(Dense(2))
model.summary()

# 3. 실행 epoch 반복 횟수
#opimizer = rmsprop, adam  / batch_size는 데이터를 자르는 크기
model.compile(optimizer=Adam(lr=1.46e-3), loss='mse')
model.fit(x_train, y_train, validation_split=0.2, epochs= 1000)


yhat = model.predict(x_data) #예측 결과
print(yhat)
print(r2_score(y_data, yhat))

sys_list = []
dia_list = []
for i in range(len(y_data)) :
    sys_list.append(yhat[i][0]-y_data[i][0])
    dia_list.append(yhat[i][1]-y_data[i][1])

plt.hist(sys_list, bins=100, density=True, histtype='step')
plt.show()
plt.hist(dia_list, bins=100, density=True, histtype='stepfilled')
plt.show()

#model.save('ppg_model.h5') #모델 저장