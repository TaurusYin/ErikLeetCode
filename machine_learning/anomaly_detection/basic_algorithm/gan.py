import plotly.graph_objects as go
import random
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# 生成时序序列
x = np.arange(1000)
y = np.random.randint(0, 10, size=(1000,))

# 注入部分异常点
for i in range(20):
    idx = random.randint(0, len(y) - 1)
    y[idx] = random.randint(20, 30)
# 对数据进行归一化
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()


# 定义生成器
def make_generator():
    model = keras.Sequential([
        keras.layers.Dense(64, input_dim=100, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# 定义判别器
def make_discriminator():
    model = keras.Sequential([
        keras.layers.Dense(256, input_dim=1, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# 定义 GAN 模型
def make_gan(generator, discriminator):
    discriminator.trainable = False
    model = keras.Sequential([
        generator,
        discriminator
    ])
    return model


generator = make_generator()
discriminator = make_discriminator()
gan = make_gan(generator, discriminator)
# 定义损失函数和优化器
loss_fn = keras.losses.BinaryCrossentropy()
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002)


# 定义训练函数
def train_gan(generator, discriminator, gan, x_train, y_train, epochs=10000, batch_size=128):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    for epoch in range(epochs):
        # 生成随机噪声
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        # 生成假数据
        fake_data = generator.predict(noise)
        # 合并真数据和假数据
        y_train_reshaped = y_train.reshape(-1, 1)
        x_combined = np.concatenate([y_train_reshaped, fake_data])
        # 创建标签
        y_combined = np.concatenate([np.ones((len(y_train), 1)), np.zeros((batch_size, 1))])
        # 训练判别器
        discriminator_loss = discriminator.train_on_batch(x_combined, y_combined)
        # 生成新的随机噪声
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        # 创建标签，全部为真（1）
        y_mislabeled = np.ones((batch_size, 1))
        # 训练生成器
        generator_loss = gan.train_on_batch(noise, y_mislabeled)
        # 打印损失
        if epoch % 100 == 0:
            print('Epoch %d: Discriminator loss: %.4f, Generator loss: %.4f' % (
                epoch, discriminator_loss, generator_loss))


train_gan(generator, discriminator, gan, x, y, epochs=1000, batch_size=128)
noise = np.random.normal(0, 1, size=(1000, 100))
fake_data = generator.predict(noise)
fake_data = scaler.inverse_transform(fake_data)
distances = np.linalg.norm(fake_data - y.reshape(-1, 1), axis=1)
print()

# 绘制生成的时序序列和异常点
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Generated Data'))
fig.add_trace(go.Scatter(x=x[distances > np.percentile(distances, 95)],
                         y=fake_data.flatten()[distances > np.percentile(distances, 95)],
                         mode='markers', marker=dict(color='red', size=8), name='Anomalies'))
fig.update_layout(title='Generated Data with Anomalies Detected', xaxis_title='Time', yaxis_title='Data')
fig.show()