"""
You are given an array of words where each word consists of lowercase English letters.

wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA without changing the order of the other characters to make it equal to wordB.

For example, "abc" is a predecessor of "abac", while "cba" is not a predecessor of "bcad".
A word chain is a sequence of words [word1, word2, ..., wordk] with k >= 1, where word1 is a predecessor of word2, word2 is a predecessor of word3, and so on. A single word is trivially a word chain with k == 1.

Return the length of the longest possible word chain with words chosen from the given list of words.



Example 1:

Input: words = ["a","b","ba","bca","bda","bdca"]
Output: 4
Explanation: One of the longest word chains is ["a","ba","bda","bdca"].
Example 2:

Input: words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
Output: 5
Explanation: All the words can be put in a word chain ["xb", "xbc", "cxbc", "pcxbc", "pcxbcf"].
Example 3:

Input: words = ["abcd","dbqca"]
Output: 1
Explanation: The trivial word chain ["abcd"] is one of the longest word chains.
["abcd","dbqca"] is not a valid word chain because the ordering of the letters is changed.

Constraints:

1 <= words.length <= 1000
1 <= words[i].length <= 16
words[i] only consists of lowercase English letters.
"""
from typing import List


class Solution:
    """
    下面是一个示例，演示在处理单词列表 ["a", "ab", "abc", "abcd"] 时，dp表的构建过程：

初始化dp表为空字典：dp = {}
遍历单词"a"，将其加入dp表，初始长度为1：dp = {"a": 1}
遍历单词"ab"，将其加入dp表，初始长度为1：dp = {"a": 1, "ab": 1}
枚举删除单词"ab"中的任意一个字符，得到"b"和"a"两个新单词：
单词"b"不在dp表中，跳过。
单词"a"在dp表中，根据dp表更新规则，将单词"ab"的长度更新为dp[a]+1=2：dp = {"a": 1, "ab": 2}
遍历单词"abc"，将其加入dp表，初始长度为1：dp = {"a": 1, "ab": 2, "abc": 1}
枚举删除单词"abc"中的任意一个字符，得到"ab"、"ac"和"bc"三个新单词：
单词"ab"在dp表中，根据dp表更新规则，将单词"abc"的长度更新为dp[ab]+1=3：dp = {"a": 1, "ab": 2, "abc": 3}
单词"ac"不在dp表中，跳过。
单词"bc"不在dp表中，跳过。
遍历单词"abcd"，将其加入dp表，初始长度为1：dp = {"a": 1, "ab": 2, "abc": 3, "abcd": 1}
枚举删除单词"abcd"中的任意一个字符，得到"abc"、"abd"、"acd"
    """
    def longestStrChain(self, words: List[str]) -> int:
        # 将单词按照长度递增排序
        words.sort(key=lambda x: len(x))
        # 构建dp表
        dp = {}
        # 遍历每个单词
        for word in words:
            # 初始化当前单词的最长单词链长度为1
            dp[word] = 1
            # 枚举删除当前单词的任意一个字符
            for i in range(len(word)):
                new_word = word[:i] + word[i+1:]
                # 如果新单词在单词列表中，则更新dp表
                if new_word in dp:
                    # 将当前单词的最长单词链长度更新为new_word的最长单词链长度加1
                    dp[word] = max(dp[word], dp[new_word]+1)
        # 返回dp表中最大值
        return max(dp.values())
