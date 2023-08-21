"""
@File    :   feature_engineering_bayes_smooth.py   
@Contact :   yinjialai 
"""
import numpy as np
from scipy.special import psi, polygamma

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha  # 初始化 alpha
        self.beta = beta    # 初始化 beta

    def update(self, impressions, clicks, iter_num=100, epsilon=1e-6):
        # 迭代更新 alpha 和 beta
        for i in range(iter_num):
            new_alpha, new_beta = self._update(impressions, clicks)
            # 当 alpha 和 beta 变化足够小，结束迭代
            if np.abs(self.alpha-new_alpha) < epsilon and np.abs(self.beta-new_beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def _update(self, impressions, clicks):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(impressions)):
            # 根据公式计算 alpha 和 beta 的分子
            numerator_alpha += (psi(clicks[i]+self.alpha) - psi(self.alpha))
            numerator_beta += (psi(impressions[i]-clicks[i]+self.beta) - psi(self.beta))
            # 计算分母
            denominator += (psi(impressions[i]+self.alpha+self.beta) - psi(self.alpha+self.beta))

        # 返回新的 alpha 和 beta
        return self.alpha*(numerator_alpha/denominator), self.beta*(numerator_beta/denominator)

# Usage:
impressions = np.array([10, 20, 30, 50, 60])  # 展示次数
clicks = np.array([1, 4, 3, 10, 15])  # 点击次数

bs = BayesianSmoothing(1, 1)  # 创建一个BayesianSmoothing对象
bs.update(impressions, clicks)  # 使用展示次数和点击次数更新参数

print(bs.alpha, bs.beta)  # 打印更新后的参数
