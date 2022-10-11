import collections
from typing import List

from sortedcontainers import SortedList

"""
https://leetcode.cn/problems/decode-string/
示例 1：

输入：s = "3[a]2[bc]"
输出："aaabcbc"
示例 2：

输入：s = "3[a2[c]]"
输出："accaccacc"
"""


def decodeString(self, s: str) -> str:
    stack, res, multi = [], "", 0
    for c in s:
        if c == '[':
            stack.append([multi, res])
            res, multi = "", 0
        elif c == ']':
            cur_multi, last_res = stack.pop()
            res = last_res + cur_multi * res
        elif '0' <= c <= '9':
            multi = multi * 10 + int(c)
        else:
            res += c
    return res


# https://leetcode.cn/problems/daily-temperatures/
def dailyTemperatures(T: List[int]) -> List[int]:
    stack = []
    res = [0] * len(T)
    for i, t in enumerate(T):
        while stack and t > T[stack[-1]]:
            res[stack.pop()] = i - stack[-1]
        stack.append(i)
    return res


temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
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
"""
找大弹大， 当前大于栈尾，递减栈； 找小弹小，当前小于栈尾，递增栈 
"""


def nextGreaterElement(nums1: List[int], nums2: List[int]) -> List[int]:
    stack = []
    res_dict = {i: -1 for i in nums2}
    for i in nums2:
        while stack and i > stack[-1]:
            small = stack.pop()
            res_dict[small] = i
        stack.append(i)
    res = []
    for j in nums1:
        res.append(res_dict[j])
    return res


nums1 = [4, 1, 2];
nums2 = [1, 3, 4, 2]
nextGreaterElement(nums1, nums2)

"""
给你一个以字符串表示的非负整数 num 和一个整数 k ，移除这个数中的 k 位数字，使得剩下的数字最小。请你以字符串形式返回这个最小的数字。
输入：num = "1432219", k = 3
输出："1219"
解释：移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219 
https://leetcode.cn/problems/remove-k-digits/solution/yi-zhao-chi-bian-li-kou-si-dao-ti-ma-ma-zai-ye-b-5/
"""


def removeKdigits(self, num, k):
    stack = []
    remain = len(num) - k
    for digit in num:
        while k and stack and digit < stack[-1]:
            stack.pop()
            k -= 1
        stack.append(digit)
    return ''.join(stack[:remain]).lstrip('0') or '0'


"""
返回 s 字典序最小的子序列，该子序列包含 s 的所有不同字符，且只包含一次。
输入：s = "bcabc"
输出："abc"
输入：s = "cbacdcbc"
输出："acdb"
"""


def smallestSubsequence(self, s: str) -> str:
    stack = list()
    for k, v in enumerate(s):
        if v in stack:
            continue
        while stack and stack[-1] > v and stack[-1] in s[k:]:
            stack.pop()
        stack.append(v)
    return ''.join(stack)


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


res = cal_min_sum(arrs=[3, 1, 2, 4])


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


nums = [1, 3, -1, -3, 5, 3, 6, 7];
k = 3
_maxSlidingWindow(nums, k)


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


"""
括号展开 + 栈
给你一个字符串表达式 s ，请你实现一个基本计算器来计算并返回它的值。
注意:不允许使用任何将字符串作为数学表达式计算的内置函数，比如 eval() 。
示例 1：
输入：s = "1 + 1"
输出：2
示例 2：
输入：s = " 2-1 + 2 "
输出：3
链接：https://leetcode.cn/problems/basic-calculator
"""


def calculate(self, s: str) -> int:
    res, num, sign = 0, 0, 1
    stack = []
    for c in s:
        if c.isdigit():
            num = 10 * num + int(c)
        elif c == "+" or c == "-":
            res += sign * num
            num = 0
            sign = 1 if c == "+" else -1
        elif c == "(":
            stack.append(res)
            stack.append(sign)
            res = 0
            sign = 1
        elif c == ")":
            res += sign * num
            num = 0
            res *= stack.pop()
            res += stack.pop()
    res += sign * num
    return res


"""
+-*/
示例 1：
输入：s = "3+2*2"
输出：7
示例 2：

输入：s = " 3/2 "
输出：1
"""


def calculate(self, s: str) -> int:
    n = len(s)
    stack = []
    preSign = '+'
    num = 0
    for i in range(n):
        if s[i] != ' ' and s[i].isdigit():
            num = num * 10 + ord(s[i]) - ord('0')
        if i == n - 1 or s[i] in '+-*/':
            if preSign == '+':
                stack.append(num)
            elif preSign == '-':
                stack.append(-num)
            elif preSign == '*':
                stack.append(stack.pop() * num)
            else:
                stack.append(int(stack.pop() / num))
            preSign = s[i]
            num = 0
    return sum(stack)


"""
common calculate
"""


def calculate(self, s: str) -> int:
    # time O(n), space O(n)
    if not s:
        return 0

    priority = {
        '(': 0,
        ')': 0,
        '+': 1,
        '-': 1,
        '*': 2,
        '/': 2
    }

    op = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b)
    }

    stack_opt = []
    stack_num = []

    s = '(' + s + ')'
    s = s.replace(' ', '').replace('(+', '(0+').replace('(-', '(0-')
    i = 0

    while i < len(s):
        if s[i] == ' ':
            i += 1
            continue

        if s[i].isdigit():
            num = 0
            while i < len(s) and s[i].isdigit():
                num = num * 10 + int(s[i])
                i += 1

            stack_num.append(num)
        elif s[i] == '(':
            stack_opt.append('(')
            i += 1
        elif s[i] == ')':
            while stack_opt and stack_opt[-1] != '(':
                operation = stack_opt.pop()
                b = stack_num.pop()
                a = stack_num.pop()
                stack_num.append(op[operation](a, b))

            # discard '('
            stack_opt.pop()
            i += 1
        else:
            # '+-'
            while stack_opt and priority[stack_opt[-1]] >= priority[s[i]]:
                operation = stack_opt.pop()
                b = stack_num.pop()
                a = stack_num.pop()
                stack_num.append(op[operation](a, b))

            stack_opt.append(s[i])
            i += 1

    return stack_num[-1]


"""
https://leetcode.cn/problems/min-stack/solution/zui-xiao-zhan-by-leetcode-solution/
"""
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = [math.inf]

    def push(self, x: int) -> None:
        self.stack.append(x)
        self.min_stack.append(min(x, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


class MaxStack:
    def __init__(self):
        self.idx, self.stk, self.sl = 0, dict(), SortedList()

    def push(self, x: int) -> None:
        self.stk[self.idx] = x
        self.sl.add((x, self.idx))
        self.idx += 1

    def pop(self) -> int:
        i, x = self.stk.popitem()
        self.sl.remove((x, i))
        return x

    def top(self) -> int:
        return next(reversed(self.stk.values()))

    def peekMax(self) -> int:
        return self.sl[-1][0]

    def popMax(self) -> int:
        x, i = self.sl.pop()
        self.stk.pop(i)
        return x
