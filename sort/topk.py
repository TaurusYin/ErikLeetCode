import collections
import heapq
from collections import Counter
from random import random
from typing import List


def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    count = collections.Counter(nums)
    num_cnt = list(count.items())
    topKs = self.findTopK(num_cnt, k, 0, len(num_cnt) - 1)
    return [item[0] for item in topKs]


def findTopK(self, num_cnt, k, low, high):
    # https://leetcode.cn/problems/top-k-frequent-elements/solution/347-qian-k-ge-gao-pin-yuan-su-zhi-jie-pa-wemd/
    pivot = random.randint(low, high)
    num_cnt[low], num_cnt[pivot] = num_cnt[pivot], num_cnt[low]
    base = num_cnt[low][1]
    i = low
    for j in range(low + 1, high + 1):
        if num_cnt[j][1] > base:
            num_cnt[i + 1], num_cnt[j] = num_cnt[j], num_cnt[i + 1]
            i += 1
    num_cnt[low], num_cnt[i] = num_cnt[i], num_cnt[low]
    if i == k - 1:
        return num_cnt[:k]
    elif i > k - 1:
        return self.findTopK(num_cnt, k, low, i - 1)
    else:
        return self.findTopK(num_cnt, k, i + 1, high)


def topKFrequent(nums, k):
    dict_count = dict(Counter(nums))
    sorted_list = sorted(dict_count.items(), key=lambda e: e[1], reverse=True)
    res = []
    for i in range(k):
        res.append(sorted_list[i][0])
    return res


def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    count = collections.Counter(nums)
    return [item[0] for item in count.most_common(k)]


def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    count = collections.Counter(nums)
    heap = [(val, key) for key, val in count.items()]
    return [item[1] for item in heapq.nlargest(k, heap)]


def topKFrequent(nums: List[int], k: int) -> List[int]:
    count = collections.Counter(nums)
    heap = []
    for key, val in count.items():
        if len(heap) >= k:
            if val > heap[0][0]:
                heapq.heapreplace(heap, (val, key))
        else:
            heapq.heappush(heap, (val, key))
    return [item[1] for item in heap]


res = topKFrequent(nums=[1, 1, 1, 2, 2, 3], k=2)

# O(N*logK)
def findKthLargest(nums: List[int], k: int) -> int:
    #   构造大小为 k 的小顶堆
    heap = [x for x in nums[:k]]
    heapq.heapify(heap)
    n = len(nums)
    for i in range(k, n):
        if nums[i] > heap[0]:
            heapq.heappop(heap)
            heapq.heappush(heap, nums[i])
    return heap[0]

#   https://leetcode.cn/problems/kth-largest-element-in-an-array/solution/pythonpython3-top-kwo-yong-si-chong-fang-y2t2/
def _findKthLargest(nums: List[int], k: int) -> int:
    heap = []
    for num in nums:
        # 如果堆长度没到k，无脑塞
        if len(heap) < k:
            heapq.heappush(heap, num)
        else:
            # 如果长度到k了，且当前元素比堆顶要大，我们才加进去，当然要先把最小的pop出来再加！
            if num > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, num)
    return heapq.heappop(heap)
findKthLargest(nums=[4, 10, 11, 29, 2, 3, 101], k=5)


def _findKthLargest(nums: List[int], k: int) -> int:
    n = len(nums)

    def quick_find(start, end):
        """
        这是类快排，大体和快排相同但做了适当优化
        """
        nonlocal n
        pivot = start
        left, right = start, end

        while left < right:
            while left < right and nums[right] >= nums[pivot]: right -= 1  # 如果当前值大于nums[pivot]的值，就继续往左找，直到发现小于的值
            while left < right and nums[left] <= nums[pivot]: left += 1
            nums[left], nums[right] = nums[right], nums[left]
        nums[pivot], nums[right] = nums[right], nums[pivot]  # 此时，pivot左边的数都比nums[pivot]小。同理，右边的都比nums[pivot]大

        if right == n - k:
            return nums[right]  # 此时说明发现了第k大的数，返回
        elif right < n - k:
            return quick_find(right + 1, end)  # 说明第k大的数在右半边，只排右边的
        else:
            return quick_find(start, right - 1)  # 同理，只排左边的

    return quick_find(0, n - 1)
