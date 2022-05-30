import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm


x_data = np.random.rand(100)
noise = np.random.normal(0,0.01,x_data.shape)
y_data = x_data * 0.2 + 0.2 + noise

# 显示为散点图
# plt.scatter(x_data,y_data)
# plt.show()

# 构建顺序模型
model = Sequential()
# 添加输出1维 输入1维的连接层
model.add(Dense(units=1,input_dim=1))
# sgd 随机梯度下降优化方法 mse 均方误差
model.compile(optimizer='sgd',loss='mse')

# 训练3001次 分批次 因为数据不是很大 干脆每批次寻来拿所有数据
for _ in tqdm(range(3000)):
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