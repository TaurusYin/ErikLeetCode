import random
from collections import Counter


class ReservoirSample(object):
    def __init__(self, size):
        self._size = size
        self._counter = 0
        self._sample = []

    def feed(self, item):
        self._counter += 1
        # 第i个元素（i <= k），直接进入池中
        if len(self._sample) < self._size:
            self._sample.append(item)
            return self._sample
        # 第i个元素（i > k），以k / i的概率进入池中
        rand_int = random.randint(1, self._counter)
        if rand_int <= self._size:
            self._sample[rand_int - 1] = item
        return self._sample

    def test_reservoir_sample(self):
        samples = []
        for i in range(10000):
            sample = []
            rs = ReservoirSample(3)
            for item in range(1, 11):
                sample = rs.feed(item)
            samples.extend(sample)
        r = Counter(samples)
        print(r)

if __name__ == '__main__':
    rs = ReservoirSample(size=3)
    rs.test_reservoir_sample()