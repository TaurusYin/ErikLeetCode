"""
@File    :   basic_timeseries.py   
@Contact :   yinjialai 
"""


def arima():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA

    # 生成数据
    np.random.seed(123)
    data = np.random.normal(0, 1, size=100)

    # 创建时间序列
    ts = pd.Series(data, index=pd.date_range(start='2000-01-01', periods=100, freq='D'))

    # 构造ARIMA模型
    model = ARIMA(ts, order=(2, 0, 2))
    """
    order : tuple, optional
        The (p,d,q) order of the model for the autoregressive, differences, and
        moving average components. d is always an integer, while p and q may
        either be integers or lists of integers.
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. Default
        is (0, 0, 0, 0). D and s are always integers, while P and Q
        may either be integers or lists of positive integers.
    """
    model_fit = model.fit()

    # 预测数据
    predictions = model_fit.predict(start=90, end=100, dynamic=False)

    # 画图
    plt.plot(ts)
    plt.plot(predictions, color='red')
    plt.show()


def kalman_filter():
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate a time series with some anomalies
    np.random.seed(0)
    x = np.linspace(0, 100, 1000)
    y = np.sin(x / 10) + np.random.normal(0, 0.1, 1000)
    y[450:550] = y[450: 550] + np.random.normal(0, 0.5, 100)

    import numpy as np

    def KalmanFilter(y, x=None, n=1, m=1, Q=0.01, R=0.1):
        """
        Implementation of Kalman Filter for time series anomaly detection

        Parameters:
        y (numpy array): The original time series
        x (numpy array): Initial estimate of the state, default is None (nx1)
        n (int): Number of state variables, default is 1
        m (int): Number of measurements, default is 1
        Q (float): Process noise covariance, default is 0.01
        R (float): Measurement noise covariance, default is 0.1

        Returns:
        x_hat (numpy array): The estimated state (len(y) x n)
        y_hat (numpy array): The predicted measurement (len(y) x m)
        residuals (numpy array): The difference between the original measurement and predicted measurement (len(y) x m)

        https://blog.csdn.net/u012554092/article/details/78290223

        """
        # Initialize state and error covariance
        if x is None:
            x = np.zeros((n, 1))
            P = np.zeros((n, n))
        else:
            P = np.eye(n)

        # Initialize arrays to store results
        x_hat = np.zeros((len(y), n))
        y_hat = np.zeros((len(y), m))
        residuals = np.zeros((len(y), m))

        # Kalman filter loop
        for i in range(len(y)):
            # Predict the measurement
            if i == 0:
                y_hat[i] = x.T @ np.ones((n, m))
            else:
                x = x_hat[i - 1].reshape(-1, 1)
                y_hat[i] = x.T @ np.ones((n, m))

            # Compute the residuals
            residuals[i] = y[i] - y_hat[i]

            # Calculate the Kalman gain
            S = P @ np.ones((n, m)) @ np.ones((m, n)) + R
            K = P @ np.ones((n, m)) @ np.linalg.inv(S)

            # Update the state estimate
            x = x + K @ residuals[i].reshape(-1, 1)

            # Update the error covariance
            P = (np.eye(n) - K @ np.ones((m, n))) @ P

            # Store the updated state estimate
            x_hat[i] = x.flatten()

            # Add process noise to the error covariance
            P = P + Q

        # Return the estimated state, predicted measurement, and residuals
        return x_hat, y_hat, residuals

    # Apply the Kalman filter
    x_hat, y_hat, residuals = KalmanFilter(y)

    # Plot the original time series and the filtered time series
    plt.plot(x, y, label='original time series')
    plt.plot(x, y_hat, label='filtered time series')
    plt.legend()
    plt.show()

    # Plot the residuals
    plt.plot(x, residuals, label='residuals')
    plt.legend()
    plt.show()

def time_domain():
    import numpy as np
    from statsmodels.graphics.tsaplots import plot_pacf
    import matplotlib.pyplot as plt

    np.random.seed(0)  # 设置随机数种子以复现相同的随机数

    # 生成周期为5的随机波形
    x = np.linspace(0, 2 * np.pi, num=100)
    y = np.sin(5 * x) + np.random.normal(0, 0.1, 100)

    # 创建图并绘制波形
    plt.plot(x, y)
    plt.show()

    # 计算自相关系数
    corr = np.correlate(y, y, mode='same') / 100


    # 创建图并绘制自相关系数
    fig, ax = plt.subplots()
    ax.plot(corr)
    ax.set_xlabel("Time Lag (samples)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of Random Time Series")
    plt.show()

    plot_pacf(y, lags=20)
    plt.title("Partial Autocorrelation Plot")
    plt.show()

    # 计算差分
    y_diff = np.diff(y)

    # 创建图并绘制差分
    fig, ax = plt.subplots()
    ax.plot(y_diff)
    ax.set_xlabel("Time")
    ax.set_ylabel("Differenced Value")
    ax.set_title("Differencing of Random Time Series")
    plt.show()


if __name__ == '__main__':
    time_domain()
    kalman_filter()
    arima()
