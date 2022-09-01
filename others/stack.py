import collections
from typing import List


# https://leetcode.cn/problems/daily-temperatures/
def dailyTemperatures(T: List[int]) -> List[int]:
    stack = []
    res = [0] * len(T)
    for i, t in enumerate(T):
        while stack and t > T[stack[-1]]:
            res[stack.pop()] = i - stack[-1]
        stack.append(i)
    return res

temperatures = [73,74,75,71,69,72,76,73]
dailyTemperatures(temperatures)

# https://leetcode.cn/problems/next-greater-element-ii/submissions/
def nextGreaterElementsCircle(self, nums: List[int]) -> List[int]:
    stack = []
    T = nums
    l = len(T)
    res = [-1] * len(T)
    T.extend(T)
    for i, t in enumerate(T):
        while stack and t > T[stack[-1]]:
            res[stack.pop() % l] = t
        stack.append(i % l)
    return res


# O(m+n) https://leetcode.cn/problems/next-greater-element-i/
def nextGreaterElement(nums1: List[int], nums2: List[int]) -> List[int]:
    stack = []
    res = [0] * len(nums2)
    hash_map = {}
    for i, t in enumerate(nums2):
        while stack and t > nums2[stack[-1]]:
            index = stack.pop()
            hash_map[nums2[index]] = t
            # res[index] = i - stack[-1]
        stack.append(i)
    print(hash_map)
    res = []
    for elem in nums1:
        if elem in hash_map:
            res.append(hash_map[elem])
        else:
            res.append(-1)
    return res
nums1 = [4,1,2]; nums2 = [1,3,4,2]
nextGreaterElement(nums1, nums2)

def get_all_sub(arr):
    n = len(arr)

    for i in range(2 ** n):
        result = []
        for j in range(n):
            if (i >> j) % 2:
                result.append(arr[j])
    return result


def cal_min_sum(arrs):
    res = 0
    stack_num = [0]
    stack_data = [[0, 0]]
    for num in arrs:
        count = 1
        if num > stack_num[-1]:
            total_sum = num + stack_data[-1][0]
        else:
            while stack_num and stack_num[-1] > num:
                count = count + stack_data[-1][1]
            stack_data.pop()
            stack_num.pop()
    total_sum = num * count + stack_data[-1][0]
    stack_data.append([total_sum, count])
    stack_num.append(num)
    res += total_sum
    return res

res = cal_min_sum(arrs = [3,1,2,4])


# https://leetcode.cn/problems/sliding-window-maximum/
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    from sortedcontainers import SortedList
    sl = SortedList(nums[:k])
    res = [max(nums[:k])]
    for i in range(k, len(nums)):
        print(nums[i])
        removed_value = nums[i - k]
        sl.remove(removed_value)
        sl.add(nums[i])
        res.append(sl[-1])
    return res

def _maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    n = len(nums)
    queue = []
    ans = []
    for i in range(n):
        while queue and nums[queue[-1]] <= nums[i]:
            queue.pop()
        queue.append(i)
        while queue[0] <= i - k:
            queue.pop(0)
        ans.append(nums[queue[0]])
    return ans[k - 1:]
nums = [1,3,-1,-3,5,3,6,7]; k = 3
_maxSlidingWindow(nums,k)



class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue1 = collections.deque()
        self.queue2 = collections.deque()


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.queue2.append(x)
        while self.queue1:
            self.queue2.append(self.queue1.popleft())
        self.queue1, self.queue2 = self.queue2, self.queue1


    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.queue1.popleft()


    def top(self) -> int:
        """
        Get the top element.
        """
        return self.queue1[0]


    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return not self.queue1


class MyQueue:

    def __init__(self):
        """
        in主要负责push，out主要负责pop
        """
        self.stack_in = []
        self.stack_out = []

    def push(self, x: int) -> None:
        """
        有新元素进来，就往in里面push
        """
        self.stack_in.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.empty():
            return None

        if self.stack_out:
            return self.stack_out.pop()
        else:
            for i in range(len(self.stack_in)):
                self.stack_out.append(self.stack_in.pop())
            return self.stack_out.pop()

    def peek(self) -> int:
        """
        Get the front element.
        """
        ans = self.pop()
        self.stack_out.append(ans)
        return ans

    def empty(self) -> bool:
        """
        只要in或者out有元素，说明队列不为空
        """
        return not (self.stack_in or self.stack_out)
