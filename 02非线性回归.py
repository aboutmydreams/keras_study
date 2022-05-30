import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from tqdm import tqdm


x_data = np.linspace(-0.5,0.5,200)
noise = np.random.normal(0,0.02,x_data.shape)
# 开平方 加噪声
y_data = np.square(x_data) + noise


# 构建顺序模型
model = Sequential()
# 1-10-1 10个隐藏层
model.add(Dense(units=10,input_dim=1))
# 添加双曲线激活函数
# model.add(Dense(units=10,input_dim=1,activation='relu'))
model.add(Activation('tanh'))
model.add(Dense(units=1))
model.add(Activation('tanh'))
# sgd 默认学习率是0.01 可以替换
sgd = SGD(lr=0.5)
model.compile(optimizer=sgd,loss='mse')


# 训练3001次 分批次 因为数据不是很大 干脆每批次寻来拿所有数据
for _ in tqdm(range(2000)):
    # train_on_batch 返回
    cost = model.train_on_batch(x_data,y_data)

w, b = model.layers[0].get_weights()
print('W:',w)
print('B:',b)
print(cost)

# 预测
y_pred = model.predict(x_data)

# 显示预测结果
plt.scatter(x_data,y_data)
plt.plot(x_data,y_pred,'r-',lw=3)


# 打印误差均值
lost = y_pred-y_data
lost = np.where(lost>=0,lost,-lost)
print(np.mean(lost))
plt.show()