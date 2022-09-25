"""
"""
import time
from collections import deque

import pandas as pd


class MovingAverage:
    def __init__(self, num_bin: int, window: float):
        self.num_bin = num_bin
        self.window = window
        self.mean = 0
        self.window_sum = 0
        self.count = 0
        self.queue = deque()
        self.pre_timestamp = None
        pass

    def Get(self) -> float:
        "返回当前计算的平均值"
        if self.queue == 0:
            return 0
        else:
            current_ts = 1360250000002.0
            head_ts = self.queue[0][0]
            tail_ts = self.queue[-1][0]
            tail_value = self.queue[-1][1]
            interval = current_ts - tail_ts
            sum_value = self.window_sum + interval * tail_value
            mean_value = sum_value / (current_ts - head_ts)
            return mean_value

    def Update(self, timestamp: float, value: float):
        "新数据到达，更新状态"
        self.count += 1
        if len(self.queue) == 0:
            prev_area = 0
        else:
            interval = timestamp - self.queue[-1][0]
            prev_value = self.queue[-1][1]
            prev_area = prev_value * interval
            self.queue[-1][2] = prev_area
        elem = [timestamp, value, None]
        self.queue.append(elem)
        head_ts, tail_ts = self.queue[0][0], self.queue[-1][0]
        if head_ts < timestamp - self.window:
            head = self.queue.popleft()
            head_area = head[2] # area = value * interval
        else:
            head_area = 0
        self.window_sum = self.window_sum - head_area + prev_area


    def MockTask(self, t):
        for id, rows in t.iterrows():
            timestamp = rows['date']
            value = rows['open']
            self.Update(timestamp, value)


if __name__ == '__main__':
    file_path = '/Users/yineric/Downloads/archive/all_stocks_5yr.csv'
    t = pd.read_csv(file_path, usecols=['date', 'open']).head(2)
    num_bin = 10
    window = 15.0
    ma = MovingAverage(num_bin=num_bin, window=window)
    start_time = time.time()
    ma.MockTask(t)
    res = ma.Get()
    end_time = time.time()
    print('time consume: {}'.format(end_time - start_time))
    print()
