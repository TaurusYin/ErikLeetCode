import time
from cmath import inf
from collections import deque
from readerwriterlock import rwlock

marker = rwlock.RWLockFair()


class MovingAverage:
    def __init__(self, num_bin: int, window: float):
        """
        :param num_bin:
        :param window:
        self.window_sum: sum value of the windows. remove head elem and append the tail elem
        self.queue: 双端队列保存窗口内的elem，elem的格式含义如下， elem: [current_timestamp, current_value, current_area]
                    current_timestamp: 当前时间戳, current_value: 当前股价, current_area: 当前股价与间隔的积分面积
                    current_area的值需要下一个elem进来之后通过该公式得出并更新: current_area = current_value * (next_timestamp - prev_timestamp)
                    由于频繁的队首队尾部操作，考虑使用双端队列，首尾操作时间复杂度O(1)
        self.window_sum: 窗口内元素current_area的总和, 维护window_sum变量用来避免频繁计算队列的和，时间复杂度可从O(n)优化至->O(1)
        self.prev_mean: 每次流数据进入一个元素，根据当前元素的时间戳，前一个元素的 prev_area = prev_value * (current_timestamp - prev_timestamp)
                        self.prev_mean = self.window_sum / (最后一个元素的时间戳 - 第一个元素的时间戳)
        """
        self.num_bin = num_bin
        self.window = window
        self.window_sum = 0
        self.queue = deque([])
        self.prev_mean = 0
        pass

    def Get(self, current_ts=None) -> float:
        "返回当前计算的平均值"
        if self.queue == 0:
            return 0
        else:
            # 线程安全加读写锁的读锁
            read_marker = marker.gen_rlock()
            read_marker.acquire()
            current_ts = time.time() * 1000 if not current_ts else current_ts
            if current_ts < self.queue[-1][0]:
                return self.prev_mean
            head_ts = self.queue[0][0]
            tail_ts = self.queue[-1][0]
            tail_value = self.queue[-1][1]
            interval = current_ts - tail_ts
            sum_value = self.window_sum + interval * tail_value
            mean_value = sum_value / (current_ts - head_ts)
            read_marker.release()
            return mean_value

    """
    1.如何淘汰W窗口头
    2.如何淘汰num_bin，把prev和curr合并，area加到前面的元素
    """

    def Update(self, timestamp: float, value: float):
        "新数据到达，更新状态"
        # 线程安全加读写锁的写锁
        write_marker = marker.gen_wlock()
        write_marker.acquire()
        """
        初始化操作：
        1. 每一次新元素到达更新前一个时间戳的股价面积积分 current_area
        2. 忘队尾部插入元素 elem = [timestamp, value, None] ， 第三个股价积分面积area初始化为None
        """
        if len(self.queue) == 0:
            prev_area = 0
        else:
            interval = timestamp - self.queue[-1][0]
            prev_value = self.queue[-1][1]
            prev_area = prev_value * interval
            # 每一次新元素到达更新前一个时间戳的股价面积积分 current_area
            self.queue[-1][2] = prev_area

        elem = [timestamp, value, None]
        self.queue.append(elem)
        """
        Window时间窗口淘汰策略:
        在窗口内判断双端队列队首的时间戳是否过期，考虑到数据粒度间隔不均匀的情况，如果直接删掉队首会造成均值突然变化比较大的情况。
        例如: 假设大部分元素都是毫秒粒度，但队首元素持续超过1个小时后，下一个元素才来，这样队首的股价积分面积
        由于前后时间戳差距变大非常巨大(current_area = current_value * (next_timestamp - prev_timestamp))
        所以分情况讨论：
        先计算溢出时间戳长度diff_ts = timestamp(当前时间戳) - self.window(窗口时间) - head_ts(队首时间戳) 
        diff_ts的长度计算需要扣除的 reduce_area (需要扣除的股价积分面积)，reduce_area = 队首元素股价 * diff_ts
        队首剩余股价积分面积 remain_area = head_area - reduce_area
        1. 如果remain_area > 0, 说明队首的积分面积还可以再减，就扣除reduce_area, 保证window_sum随着新元素到来一点一点扣除
                               扣减后队首的时间戳相应平移 diff_ts 长度, 不断扣减直到队首的时间戳不过期位置
        2. 如果remain_area <=0, 说明队首的积分面积不够扣减，则直接弹出队首元素，将整个剩余股价积分面积扣减掉
        self.window_sum = self.window_sum - total_reduce_area + prev_area
        """
        head_ts, tail_ts = self.queue[0][0], self.queue[-1][0]
        if head_ts < timestamp - self.window:
            total_reduce_area = 0
            while head_ts < timestamp - self.window:
                diff_ts = timestamp - self.window - head_ts
                reduce_area = self.queue[0][1] * diff_ts
                if self.queue[0][2] - reduce_area > 0:
                    self.queue[0][2] -= reduce_area  # 队首股价积分面积需要扣除溢出长度的股价积分面积
                    self.queue[0][0] += diff_ts  # 扣减后队首的时间戳相应平移 diff_ts 长度，保证队列中元素都在window范围内
                else:
                    head = self.queue.popleft()
                    reduce_area = head[2]  # area_i = value_i * interval
                total_reduce_area += reduce_area
                head_ts = self.queue[0][0]
        else:
            total_reduce_area = 0

        self.window_sum = self.window_sum - total_reduce_area + prev_area
        total = sum(list(map(lambda x: x[2], self.queue))[:-1])
        if abs(total - self.window_sum) > 0.000001:
            print()

        if len(self.queue) <= 1:
            self.prev_mean = self.queue[0][1]
        else:
            self.prev_mean = self.window_sum / (self.queue[-1][0] - self.queue[0][0])

        """
        内存不足合并策略：
        
        """
        if len(self.queue) >= self.num_bin:
            diff = inf
            for idx, item in enumerate(self.queue):
                if idx >= 1 and idx < len(self.queue) - 1:
                    timestamp, value, area = item[0], item[1], item[2]
                    prev_timestamp, prev_value, prev_area = self.queue[idx - 1][0], self.queue[idx - 1][1], \
                                                            self.queue[idx - 1][2]
                    cur_diff = abs(value - self.prev_mean)
                    if cur_diff < diff:
                        diff = cur_diff
                        elem = [timestamp, value, area]
                        prev_elem = [prev_timestamp, prev_value, prev_area]

            self.queue.remove(elem)
            self.window_sum -= elem[2]
            self.prev_mean = self.window_sum / (self.queue[-1][0] - self.queue[0][0])
            print()

        write_marker.release()

    def MockTask(self, mock_data):
        for timestamp, value in mock_data:
            self.Update(timestamp, value)
