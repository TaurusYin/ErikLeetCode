"""
Write an algorithm to determine if a number n is happy.

A happy number is a number defined by the following process:

Starting with any positive integer, replace the number by the sum of the squares of its digits.
Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
Those numbers for which this process ends in 1 are happy.
Return true if n is a happy number, and false if not.


Example 1:

Input: n = 19
Output: true
Explanation:
1*1 + 9*9 = 82
8*8 + 2*2 = 68
6*6 + 8*8 = 100
1*1 + 0*0 + 0*0 = 1
Example 2:

Input: n = 2
Output: false

Constraints:

1 <= n <= 231 - 1
"""


class Solution:
    def isHappy(self, n: int) -> bool:
        """
        这种算法的关键在于如何判断一个链表是否带环。我们使用两个指针，一个慢指针每次走一步，一个快指针每次走两步，如果这个链表带环，那么快指针最终一定会追上慢指针。这是因为如果快指针和慢指针分别走了 k 步和 2k 步，那么它们之间的距离就是 k 步，每次快指针比慢指针多走一步，这个距离就会减少一步，当距离减少到 0 的时候，快指针就追上了慢指针。如果链表不带环，那么快指针最终会指向链表的末尾节点，此时算法结束。
        该题目可以使用快慢指针的思路来解决，如果一个数是 happy number，那么最终计算的结果一定是 1。如果不是 happy number，则最终结果会进入一个循环，即结果会在某个数字处循环出现。我们可以用快指针每次计算两步，慢指针每次计算一步，如果最终结果是 1，那么快指针一定会先到达 1，如果最终结果不是 1，那么快指针和慢指针最终一定会相遇。

        具体思路如下：
定义快慢指针，初始值都为 n。对于每次循环，慢指针计算一次数字的平方和，快指针计算两次数字的平方和。
如果最终结果是 1，返回 true。如果最终结果不是 1，快指针和慢指针会相遇，返回 false。
        :param n:
        :return:
        """
        slow = fast = n
        while True:
            slow = sum(int(i) ** 2 for i in str(slow))
            fast = sum(int(i) ** 2 for i in str(fast))
            fast = sum(int(i) ** 2 for i in str(fast))
            if slow == fast:
                break
        return slow == 1


final_res = Solution().isHappy(n=2)
print()
