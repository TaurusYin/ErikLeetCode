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
        self.timestamp: 当前元素毫秒时间戳
        self.prev_area: 前一个元素的股价积分面积
        """
        self.num_bin = num_bin
        self.window = window
        self.window_sum = 0
        self.queue = deque([])
        self.prev_mean = 0
        self.timestamp = None
        self.prev_area = None

    def get(self, current_ts: float = None) -> float:
        """
        由于均值通过股价积分面积求得，所以需要知道当前时间戳，interval = 当前时间戳-前一个时间戳
        股价积分面积 = 前一个元素的股价 * interval ,即不同的当前时间戳均值会随传入的当前时间戳的变化而变化
        当知道了interval就可以算出，前一个时间戳的元素实际的股价积分面积，此方法解决了考虑数据间隔变化极端的情况下的鲁棒性
        """
        try:
            if len(self.queue) == 0:
                return 0
            else:
                # 线程安全加读写锁的读锁
                read_marker = marker.gen_rlock()
                read_marker.acquire()
                current_ts = time.time() * 1000 if not current_ts else current_ts
                head_ts, tail_ts, tail_value = self.queue[0][0], self.queue[-1][0], self.queue[-1][1]
                interval = current_ts - tail_ts
                sum_value = self.window_sum + interval * tail_value
                mean_value = sum_value / (current_ts - head_ts)
                read_marker.release()
                return mean_value
        except Exception as e:
            print(e)

    def update(self, timestamp: float, value: float):
        # 线程安全加读写锁的写锁
        try:
            write_marker = marker.gen_wlock()
            write_marker.acquire()
            # 默认入队的timestamp比队尾的timestamp要大
            # 初始化计算前一个元素的股价积分面积并更新
            self.calculate_prev_area(timestamp=timestamp, value=value)
            # 时间窗口淘汰策略
            self.window_strategy(timestamp=timestamp)
            # 内存空间合并策略
            self.memory_strategy()
            write_marker.release()
        except Exception as e:
            print(e)

    def calculate_prev_area(self, timestamp: float, value: float):
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
        self.prev_area = prev_area

    def window_strategy(self, timestamp: float):
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

        self.window_sum = self.window_sum - total_reduce_area + self.prev_area

    def memory_strategy(self):
        """
        内存不足合并策略：找队列中相邻元素股价最相近(甚至相等)的元素，把股价相近的股价积分面积合并。合并元素并不影响窗口sum值
        这样等该合并的元素在Window窗口淘汰队首的时候，即使该元素的时间戳跨度由于合并后比较大，但是由于合并的股价接近，
        所以逐步扣减淘汰过程中，窗口内的均值误差很小，(如果合并的是股价相等的元素，误差为0)
        双端队列删除元素时间复杂度为O(n), 查找相差最小的相邻元素为O(n)
        """
        global prev_index, elem
        if len(self.queue) >= self.num_bin:
            diff = inf
            for idx, item in enumerate(self.queue):
                if 1 <= idx < len(self.queue) - 1:
                    timestamp, value, area = item[0], item[1], item[2]
                    prev_timestamp, prev_value, prev_area = self.queue[idx - 1][0], self.queue[idx - 1][1], \
                                                            self.queue[idx - 1][2]
                    cur_diff = abs(value - prev_value)
                    if cur_diff < diff:
                        diff = cur_diff
                        elem = [timestamp, value, area]
                        prev_index = idx - 1
            # 更新elem元素的前一个元素的股价积分面积，合并两个元素的股价积分面积
            self.queue[prev_index][2] += elem[2]
            # 更新elem元素的前一个元素的平均股价 = 合并后股价积分面积 /（elem的后一个元素时间戳 - elem的前一个元素时间戳）
            self.queue[prev_index][1] = self.queue[prev_index][2] / (
                    self.queue[prev_index + 2][0] - self.queue[prev_index][0])
            # 删除当前elem元素
            self.queue.remove(elem)

    # 模拟流任务
    def mock_task(self, mock_data):
        for timestamp, value in mock_data:
            self.update(timestamp, value)
