import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.regularizers import l2
from tqdm import tqdm

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print('x_shape: ',x_train.shape)
print('y_shape: ',y_train.shape)
x_train = x_train.reshape(x_train.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0],-1)
# one-hot编码
y_train = np_utils.to_categorical(y_train,num_classes=10)/255
y_test = np_utils.to_categorical(y_test,num_classes=10)/255

# 创建模型 输入784 输出10 偏置bias_initializer = 1 l2正则化
model = Sequential([
    Dense(units=200,input_dim=784,bias_initializer='one',activation='tanh',bias_regularizer=l2(0.0003)),
    Dropout(0.15),# 丢失部分神经元 减少过拟合现象
    Dense(units=100,bias_initializer='one',activation='tanh',bias_regularizer=l2(0.0003)),
    Dropout(0.15),
    Dense(units=10,bias_initializer='one',activation='softmax',bias_regularizer=l2(0.0003))
])

# 定义优化器
sgd = SGD(lr=0.2)
adam = Adam(lr=0.002)
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    # loss='mse',# 均方差
    metrics=['accuracy'],
)

# 训练模型 batch_size 每批次训练数据量 epochs 迭代周期
model.fit(x_train,y_train,batch_size=32,epochs=10)
# 评估模型
loss,accuracy = model.evaluate(x_test,y_test)
print('loss:',loss)
print('accuracy:',accuracy)