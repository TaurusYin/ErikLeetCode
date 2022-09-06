"""
示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
示例 2：

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
     注意，你可以重复使用字典中的单词。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/word-break
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    import functools
    @functools.lru_cache(None)
    def back_track(s):
        if (not s):
            return True
        res = False
        for i in range(1, len(s) + 1):
            if (s[:i] in wordDict):
                res = back_track(s[i:]) or res
        return res

    return back_track(s)
