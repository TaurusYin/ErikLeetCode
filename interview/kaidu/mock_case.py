import time
import unittest

from interview.kaidu.ma_solution import MovingAverage

mock_data = [
    (1360250000000.00, 15.07),
    # (1360250000001.00, 14.89),
    # (1360250000002.00, 14.45),
    # (1360250000003.00, 14.3),
    # (1360250000004.00, 14.94),
    (1360250000005.00, 13.93),
    (1360250000006.00, 14.33),
    (1360250000007.00, 14.17),
    (1360250000008.00, 13.62),
    (1360250000009.00, 13.57),
    (1360250000010.00, 13.6),
    (1360250000011.00, 13.14),
    (1360250000012.00, 13.28),
    (1360250000013.00, 13.49),
    (1360250000014.00, 13.37),
    (1360250000015.00, 13.5),
    (1360250000016.00, 14.01),
    (1360250000017.00, 14.52),
    (1360250000018.00, 14.7),
    (1360250000019.00, 14.99),
    (1360250000020.00, 14.85),
    (1360250000021.00, 15.14),
    (1360250000022.00, 15.54),
    (1360250000023.00, 15.98),
    # (1360250000024.00, 16.45),
    # (1360250000025.00, 15.8),
    # (1360250000026.00, 16.48),
    # (1360250000027.00, 17.13),
    # (1360250000028.00, 17.21),
    # (1360250000029.00, 17.1),
    (1360250000030.00, 16.92)
]


class CaseTest(unittest.TestCase):
    # 测试Window窗口内sum值,用于测试队窗口移动过程中，过期元素时序粒度不均匀与插入元素时序粒度不均匀两种情况
    def _test_window_sum(self):
        num_bin = 100
        window = 15.0
        ma = MovingAverage(num_bin=num_bin, window=window)
        # 取最终窗口的元素，从第11个元素开始到倒数第二个元素，排除最后一个元素是因为最后一个元素不参与积分面积计算)
        actual_window_sum = sum(list(map(lambda x: x[1], mock_data[11:-1])))
        # 按照前一个股价插值, 补了6个值
        actual_window_sum += 15.98 * (1360250000030.00 - 1360250000023.00 - 1)
        ma.MockTask(mock_data=mock_data)
        # 对比实际值与窗口值
        assert abs(ma.window_sum - actual_window_sum) < 0.000000000001
        self.assertAlmostEqual(ma.window_sum, actual_window_sum)

    # 测试Window窗口内mean值
    def _test_mean(self):
        num_bin = 100
        window = 15.0
        ma = MovingAverage(num_bin=num_bin, window=window)
        # 取最终窗口的元素，从第11个元素开始到倒数第二个元素，排除最后一个元素是因为最后一个元素不参与积分面积计算)
        actual_window_sum = sum(list(map(lambda x: x[1], mock_data[11:-1])))
        # 按照前一个股价插值, 补了6个值
        actual_window_sum += 15.98 * (1360250000030.00 - 1360250000023.00 - 1)
        actual_window_mean = actual_window_sum / (len(mock_data[11:-1]) + 6)
        ma.MockTask(mock_data=mock_data)
        assert abs(ma.prev_mean - actual_window_mean) < 0.000000000001
        self.assertAlmostEqual(ma.prev_mean, actual_window_mean)

    def test_num_bin(self):
        num_bin = 10
        window = 15.0
        ma = MovingAverage(num_bin=num_bin, window=window)
        ma.MockTask(mock_data=mock_data)
        print()

if __name__ == '__main__':
    ct = CaseTest()
    ct._test_num_bin()


