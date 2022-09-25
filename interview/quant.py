"""
"""
import time
from collections import deque
from readerwriterlock import rwlock

marker = rwlock.RWLockFair()


class MovingAverage:
    def __init__(self, num_bin: int, window: float):
        """
        :param num_bin:
        :param window:
        window_sum: sum value of the windows. remove head elem and append the tail elem
        queue: head --> tail, if head.timestamp < tail.timestamp - window: remove(head)
        """
        self.num_bin = num_bin
        self.window = window
        self.window_sum = 0
        self.queue = deque()
        self.pre_timestamp = None
        pass

    def Get(self, current_ts=None) -> float:
        "返回当前计算的平均值"
        read_marker = marker.gen_rlock()
        read_marker.acquire()
        if self.queue == 0:
            mean_value = 0
        else:
            current_ts = time.time() * 1000 if not current_ts else current_ts
            head_ts = self.queue[0][0]
            tail_ts = self.queue[-1][0]
            tail_value = self.queue[-1][1]
            interval = current_ts - tail_ts
            sum_value = self.window_sum + interval * tail_value
            mean_value = sum_value / (current_ts - head_ts)
        read_marker.release()
        return mean_value

    def Update(self, timestamp: float, value: float):
        "新数据到达，更新状态"
        write_marker = marker.gen_wlock()
        write_marker.acquire()
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
            head_area = head[2]  # area = value * interval
        else:
            head_area = 0
        self.window_sum = self.window_sum - head_area + prev_area
        write_marker.release()

    def MockTask(self, mock_data):
        for timestamp, value in mock_data:
            self.Update(timestamp, value)
        '''
        for id, rows in t.iterrows():
            timestamp = rows['date']
            value = rows['open']
            self.Update(timestamp, value)
        '''
