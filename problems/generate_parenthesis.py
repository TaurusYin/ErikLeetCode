"""
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。
 

示例 1：

输入：s = "()"
输出：true
示例 2：
输入：s = "()[]{}"
输出：true

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/valid-parentheses
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""
def isValid(self, s: str) -> bool:
    if len(s) % 2 == 1:
        return False

    pairs = {
        ")": "(",
        "]": "[",
        "}": "{",
    }
    stack = list()
    for ch in s:
        if ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        else:
            stack.append(ch)

    return not stack

"""
数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
方法一还有改进的余地：我们可以只在序列仍然保持有效时才添加 \text{`('}‘(’ 或 \text{`)'}‘)’，而不是像 方法一 那样每次添加。我们可以通过跟踪到目前为止放置的左括号和右括号的数目来做到这一点，

如果左括号数量不大于 nn，我们可以放一个左括号。如果右括号数量小于左括号的数量，我们可以放一个右括号。

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/generate-parentheses/solution/gua-hao-sheng-cheng-by-leetcode-solution/
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
输入：n = 1
输出：["()"]
"""


def generateParenthesis(self, n: int) -> List[str]:
    ans = []

    def backtrack(S, left, right):
        if len(S) == 2 * n:
            ans.append(''.join(S))
            return
        if left < n:
            S.append('(')
            backtrack(S, left + 1, right)
            S.pop()
        if right < left:
            S.append(')')
            backtrack(S, left, right + 1)
            S.pop()

    backtrack([], 0, 0)
    return ans


"""
任何一个括号序列都一定是由 \text{`('}‘(’ 开头，并且第一个 \text{`('}‘(’ 一定有一个唯一与之对应的 \text{`)'}‘)’。这样一来，每一个括号序列可以用 (a)b(a)b 来表示，其中 aa 与 bb 分别是一个合法的括号序列（可以为空）。

那么，要生成所有长度为 2n2n 的括号序列，我们定义一个函数 \textit{generate}(n)generate(n) 来返回所有可能的括号序列。那么在函数 \textit{generate}(n)generate(n) 的过程中：

我们需要枚举与第一个 \text{`('}‘(’ 对应的 \text{`)'}‘)’ 的位置 2i + 12i+1；
递归调用 \textit{generate}(i)generate(i) 即可计算 aa 的所有可能性；
递归调用 \textit{generate}(n - i - 1)generate(n−i−1) 即可计算 bb 的所有可能性；
遍历 aa 与 bb 的所有可能性并拼接，即可得到所有长度为 2n2n 的括号序列。
为了节省计算时间，我们在每次 \textit{generate}(i)generate(i) 函数返回之前，把返回值存储起来，下次再调用 \textit{generate}(i)generate(i) 时可以直接返回，不需要再递归计算。

卡特兰数 (1/n+1)*(2n/n)
链接：https://leetcode.cn/problems/generate-parentheses/solution/gua-hao-sheng-cheng-by-leetcode-solution/
"""

def generateParenthesis(self, n: int) -> List[str]:
    ans = []
    def backtrack(S, left, right):
        if len(S) == 2 * n:
            ans.append(''.join(S))
            return
        if left < n:
            S.append('(')
            backtrack(S, left + 1, right)
            S.pop()
        if right < left:
            S.append(')')
            backtrack(S, left, right + 1)
            S.pop()

    backtrack([], 0, 0)
    return ans


@lru_cache(None)
def generateParenthesis(self, n: int) -> List[str]:
    if n == 0:
        return ['']
    ans = []
    for c in range(n):
        for left in self.generateParenthesis(c):
            for right in self.generateParenthesis(n - 1 - c):
                ans.append('({}){}'.format(left, right))
    return ans

"""
给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
https://leetcode.cn/problems/longest-valid-parentheses/solution/pythonjian-ji-ban-11xing-he-xin-dai-ma-by-java_lee/
"""
def longestValidParentheses(self, s: str) -> int:
    ans = 0  # 最大合法长度(返回值)
    stack = [-1, ]  # stack[0]:合法括号起点-1 ; stack[1:]尚未匹配左括号下标
    for i, ch in enumerate(s):
        if '(' == ch:  # 左括号
            stack.append(i)
        elif len(stack) > 1:  # 右括号，且有成对左括号
            stack.pop()  # 成对匹配
            ans = max(ans, i - stack[-1])
        else:  # 非法的右括号
            stack[0] = i
    return ans

"""
给你一个由数字和运算符组成的字符串 expression ，按不同优先级组合数字和运算符，计算并返回所有可能组合的结果。你可以 按任意顺序 返回答案。
生成的测试用例满足其对应输出值符合 32 位整数范围，不同结果的数量不超过 104 。
输入：expression = "2-1-1"
输出：[0,2]
解释：
((2-1)-1) = 0 
(2-(1-1)) = 2
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/different-ways-to-add-parentheses
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""
def diffWaysToCompute(self, input: str) -> List[int]:
    # 如果只有数字，直接返回
    if input.isdigit():
        return [int(input)]

    res = []
    for i, char in enumerate(input):
        if char in ['+', '-', '*']:
            # 1.分解：遇到运算符，计算左右两侧的结果集
            # 2.解决：diffWaysToCompute 递归函数求出子问题的解
            left = self.diffWaysToCompute(input[:i])
            right = self.diffWaysToCompute(input[i + 1:])
            # 3.合并：根据运算符合并子问题的解
            for l in left:
                for r in right:
                    if char == '+':
                        res.append(l + r)
                    elif char == '-':
                        res.append(l - r)
                    else:
                        res.append(l * r)

    return res
