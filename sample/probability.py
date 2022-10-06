import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="/home/ppp/anaconda3/envs/python-3.6/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/simhei.ttf")
import seaborn as sns

class PowerSupplySystem:
    def __init__(self, s0=0, t0=0.0, N=int(1e5)):
        self.N = N
        self.charging_time = 2.5
        self.s0 = s0
        self.t0 = t0
        self.reset()

    def reset(self):
        self.t = self.t0
        self.s = self.s0
        self.rs = np.random.randint(1, 7, self.N)
        self.i = 0

    def exhaust_time(self):
        nxt_t = self.rs[self.i]
        self.i += 1
        return nxt_t

    def __call__(self):
        np.random.seed()
        self.t += self.exhaust_time()
        self.s = 1
        while self.s != 2 and self.i < self.N:
            t = self.exhaust_time()
            self.t += t
            if t < self.charging_time:
                self.s = 2
        tmp = self.t
        self.reset()
        return tmp


if __name__ == '__main__':
    system = PowerSupplySystem()
    T = int(1e5)
    ts = np.zeros(T)
    for i in range(T):
        ts[i] = system()

    avg_ts = np.zeros(T)
    avg_ts[0] = ts[0]
    for i in range(1, ts.shape[0]):
        avg_ts[i] = (avg_ts[i - 1] * i + ts[i]) / (i + 1)

    print("平均停止供电时间: {:.6f}".format(avg_ts[-1]))

