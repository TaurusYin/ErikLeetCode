import collections
import functools
from functools import cache
from typing import List


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


class Solutions:
    """
    输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
链接：https://leetcode.cn/problems/word-break
    """

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordDict = set(wordDict)  # 用O(n)的时间转为哈希表，这样查询只需要O(1)

        @cache
        def backtrack(s):
            if not s: return True
            for i in range(len(s)):
                if s[:i + 1] in wordDict and backtrack(s[i + 1:]):
                    return True
            return False

        return backtrack(s)

    class Solution:
        def wordBreak(self, s: str, wordDict: List[str]) -> bool:
            wordDict = set(wordDict)  # 用O(n)的时间转为哈希表，这样查询只需要O(1)
            method = dict()  # memo

            def backtrack(s):
                if s in method: return method[s]  # memo
                if not s: return True
                for i in range(len(s)):
                    if s[:i + 1] in wordDict and backtrack(s[i + 1:]):
                        method[s] = True  # memo
                        return True
                method[s] = False
                return False

            return backtrack(s)

    """
    输入:s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
    输出:["cats and dog","cat sand dog"]
    链接：https://leetcode.cn/problems/word-break-ii
    https://leetcode.cn/problems/word-break-ii/solution/cong-hui-su-dao-di-qia-er-ji-ji-yi-hua-di-gui-140-/
    """

    def wordBreakII(self, s: str, wordDict: List[str]) -> List[str]:
        wordDict = set(wordDict)  # 用O(n)的时间转为哈希表，这样查询只需要O(1)

        def backtrack(s):
            if not s:
                res.append(' '.join(path[:]))
                return True
            for i in range(len(s)):
                if s[:i + 1] in wordDict:
                    path.append(s[:i + 1])
                    backtrack(s[i + 1:])
                    path.pop()
                    # return True
            return False

        path = []
        res = []
        backtrack(s)
        print(res)
        return res
