import collections
from collections import defaultdict
from typing import List


def diff(nums):
    n = len(nums)
    diff = [0] * n
    diff[0] = nums[0]
    for i in range(n - 1):
        diff[i + 1] = nums[i + 1] - nums[i]
    return diff


def inc(diff_nums, start, end, val):
    diff_nums[start] += val
    if end + 1 < len(diff_nums):
        diff_nums[end] -= val
    return diff_nums


def inc_dict(diff_nums, start, end, val):
    diff_nums[start] += val
    diff_nums[end] -= val
    return diff_nums


def diff_to_res(diff_nums):
    res = [0] * len(diff_nums)
    res[0] = diff_nums[0]
    for i in range(1, len(diff_nums)):
        res[i] = res[i - 1] + diff_nums[i]
    return res


class Solution:
    # https://leetcode.cn/problems/car-pooling/submissions/
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        """
        trips = [[2,1,5],[3,3,7]], capacity = 4
        给定整数capacity和一个数组 trips , trip[i] = [numPassengersi, fromi, to
        表示第 i 次旅行numPassengers乘客，接他们和放他们的位置分别是fromi和toi。这些位置是从汽车的初始位置向东的公里数。
        :param trips:
        :param capacity:
        :return:
        """
        dp = [0] * 1000
        for trip in trips:
            num_passengers, start, end = trip[0], trip[1], trip[2]
            dp = inc(dp, start, end, num_passengers)
        res = diff_to_res(dp)
        print(res)
        return capacity >= max(res)

    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        # https://leetcode.cn/problems/corporate-flight-bookings/submissions/
        diff_nums = [0] * n
        for book in bookings:
            first, last, seats = book[0] - 1, book[1] - 1, book[2]
            diff_nums = inc(diff_nums, first, last + 1, seats)
        res = diff_to_res(diff_nums)
        return res

    def _minMeetingRooms(self, intervals: List[List[int]]) -> int:
        # https://leetcode.cn/problems/meeting-rooms-ii/
        diff_nums = defaultdict(int)
        for start, end in intervals:
            diff_nums = inc_dict(diff_nums=diff_nums, start=start, end=end, val=1)
        cur = ans = 0
        diff_nums = sorted(diff_nums.items(), key=lambda d: d[0])
        for _, v in diff_nums.items():
            cur += v
            ans = max(ans, cur)
        return ans


if __name__ == '__main__':
    s = Solution()
    s._minMeetingRooms(intervals=[[0,30],[5,10],[15,20]])
    x = diff([8, 2, 6, 3, 1])
    y = diff_to_res(x)
    print()
