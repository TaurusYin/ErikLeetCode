"""
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
https://leetcode.cn/problems/trapping-rain-water/solution/wei-en-tu-jie-fa-zui-jian-dan-yi-dong-10xing-jie-j/
"""
import heapq
from typing import List


# 双指针 O(n) O(1),  左小移动左， 右小移动右
def trap(self, height: List[int]) -> int:
    ans = 0
    left, right = 0, len(height) - 1
    leftMax = rightMax = 0
    # 不需要相等,相等时,同时指向最大值,在最高点不会有雨水的
    while left < right:
        leftMax = max(leftMax, height[left])
        rightMax = max(rightMax, height[right])
        if height[left] < height[right]:
            ans += leftMax - height[left]
            left += 1
        else:
            ans += rightMax - height[right]
            right -= 1

    return ans


# 面积
def trap(self, height: List[int]) -> int:
    n = len(height)
    # 同时从左往右和从右往左计算有效面积
    s1, s2 = 0, 0
    max1, max2 = 0, 0
    for i in range(n):
        if height[i] > max1:
            max1 = height[i]
        if height[n - i - 1] > max2:
            max2 = height[n - i - 1]
        s1 += max1
        s2 += max2
    # 积水面积 = S1 + S2 - 矩形面积 - 柱子面积
    res = s1 + s2 - max1 * len(height) - sum(height)
    return res


# 单调栈压缩版
"""
单调栈是按照 行 的方向来计算雨水
        从栈顶到栈底的顺序：从小到大
        通过三个元素来接水：栈顶，栈顶的下一个元素，以及即将入栈的元素
        雨水高度是 min(凹槽左边高度, 凹槽右边高度) - 凹槽底部高度
        雨水的宽度是 凹槽右边的下标 - 凹槽左边的下标 - 1（因为只求中间宽度）
链接：https://leetcode.cn/problems/trapping-rain-water/solution/42-jie-yu-shui-shuang-zhi-zhen-dong-tai-wguic/
"""


def trap(self, height: List[int]) -> int:
    ans = 0
    stack = list()  # 用列表来模拟实现栈
    n = len(height)

    for i, h in enumerate(height):  # 同时获取下标和对应的高度
        # height[i] > height[stack[-1]]得到一个可以接雨水的区域
        while stack and h > height[stack[-1]]:
            top = stack.pop()
            if not stack:  # 栈为空,左边不存在最大值,无法接雨水
                break
            left = stack[-1]  # left成为新的栈顶元素
            currWidth = i - left - 1  # 获取接雨水区域的宽度
            currHeight = min(height[left], height[i]) - height[top]
            ans += currWidth * currHeight
        stack.append(i)  # 在对下标i处计算能接的雨水量之后,将i入栈,继续遍历后面的下标

    return ans


"""
动归 O(n), O(n)
"""


def trap(self, height: List[int]) -> int:
    if not height:
        return 0

    n = len(height)
    leftMax = [height[0]] + [0] * (n - 1)  # 简化的连续赋值
    # 正向遍历数组 height 得到数组 leftMax 的每个元素值
    for i in range(1, n):
        leftMax[i] = max(leftMax[i - 1], height[i])
    # 反向遍历数组 height 得到数组 rightMax 的每个元素值
    rightMax = [0] * (n - 1) + [height[n - 1]]
    for i in range(n - 2, -1, -1):  # 逆序遍历
        rightMax[i] = max(rightMax[i + 1], height[i])
    # 遍历每个下标位置即可得到能接的雨水总量
    ans = sum(min(leftMax[i], rightMax[i]) - height[i] for i in range(n))
    return ans

"""
https://leetcode.cn/problems/trapping-rain-water-ii/solution/mu-tong-yuan-li-by-fromdtor-ne9g/
先将水桶的边，最外一圈放入小根堆
然后每次从小根堆里拿出最小元素，用邻接的从没有加入小跟堆的元素替换
如果替换的板（更里面的）比原来的板高，这补齐了最短的那一块
如果替换的板（更里面的）比原来的板底，新来的板产生了蓄水，水量就是他们的差值，当然桶边板没有因此变短
所有位置都访问过以后，结束
"""
def trapRainWater(self, heightMap: List[List[int]]) -> int:
    heap = []
    visited = set()
    for i, row in enumerate(heightMap):
        for j, cell in enumerate(row):
            if i == 0 or j == 0 or i == len(heightMap) - 1 or j == len(row) - 1:
                heapq.heappush(heap, (cell, i, j))
                visited.add((i, j))

    result = 0
    while heap:
        mi, x, y = heapq.heappop(heap)
        for di in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            i, j = x + di[0], y + di[1]
            if i < 0 or i >= len(heightMap): continue
            if j < 0 or j >= len(heightMap[x]): continue
            if (i, j) not in visited:
                visited.add((i, j))
                heapq.heappush(heap, (max(mi, heightMap[i][j]), i, j))
                if heightMap[i][j] < mi:
                    result += mi - heightMap[i][j]
    return result
