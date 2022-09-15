# https://labuladong.github.io/algo/2/20/27/
from cmath import inf
from collections import OrderedDict, defaultdict, Counter
from typing import List

hash_map = Counter()
hash_map.most_common()

def sliding_windows(s):
    windows, needs = OrderedDict(), OrderedDict()
    left = right = 0
    for right in range(len(s)):
        right_elem = s[right]
        print(right_elem)
        windows.update()
        while "windows needs shrink":
            left_elem = s[left]
            left += 1
            windows.update()


s = "cbaebabacd";
p = "abc"
s = "baa";
p = "aa"


# O(n+m+Σ) 其中Σ为所有可能的字符数 https://leetcode.cn/problems/find-all-anagrams-in-a-string/
def findAnagrams(s: str, p: str) -> List[int]:
    windows, needs = Counter(), Counter(p)
    left = right = valid = 0
    res = []
    while right < len(s):
        right_elem = s[right]
        right += 1
        if right_elem in needs:
            windows[right_elem] += 1  # 先加后判断
            if windows[right_elem] == needs[right_elem]:
                valid += 1
        while right - left >= len(p):
            if valid == len(needs):
                res.append(left)
            left_elem = s[left]
            left += 1
            if left_elem in needs:
                if windows[left_elem] == needs[left_elem]:
                    valid -= 1
                windows[left_elem] -= 1  # 先判断后减

    return res


res = findAnagrams(s, p)

s1 = "ab"; s2 = "eidbaooo"
# https://leetcode.cn/problems/permutation-in-string/
def checkInclusion(self, s1: str, s2: str) -> bool:
    windows, needs = Counter(), Counter(s1)
    left = right = valid = 0
    while right < len(s2):
        right_elem = s2[right]
        right += 1
        if right_elem in needs:
            windows[right_elem] += 1
            if windows[right_elem] == needs[right_elem]:
                valid += 1
        while right - left >= len(s1):
            if valid == len(needs):
                return True
            left_elem = s2[left]
            left += 1
            if left_elem in needs:
                if windows[left_elem] == needs[left_elem]:
                    valid -= 1
                windows[left_elem] -= 1
    print(valid)
    return False


s = "ADOBECODEBANC"
t = "ABC"
s = "abc"
t = "cba"


# https://leetcode.cn/problems/minimum-window-substring/
def minWindow(s: str, t: str) -> str:
    def is_valid(windows, needs):
        valid = 0
        for key in needs.keys():
            if key in windows and windows[key] >= needs[key]:
                valid += 1
        return valid == len(needs)

    windows, needs = Counter(), Counter(t)
    left = right = 0
    m = len(s)
    max_res_len = inf
    res = None
    while right < m:
        right_elem = s[right]
        right += 1
        print(right_elem)
        if right_elem in needs.keys():
            windows[right_elem] += 1

        while is_valid(windows, needs):
            if right - left < max_res_len:
                max_res_len = right - left
                res = (right, left)

            left_elem = s[left]
            if left_elem in windows:
                windows[left_elem] -= 1
            left += 1
    return s[res[1]:res[0]] if res else ""


res = minWindow(s, t)
print()

s = "abcabcbb"
s = "bbbbb"
s = "pwwkew"
s = "au"
s = "dvdf"
s = " "


# https://leetcode.cn/problems/longest-substring-without-repeating-characters/
def lengthOfLongestSubstring(s: str) -> int:
    left = right = 0
    windows = Counter()
    n = len(s)
    max_length = 0
    while right < n:
        right_elem = s[right]
        windows[right_elem] += 1
        right += 1
        while windows[right_elem] > 1 and left < right:
            left_elem = s[left]
            windows[left_elem] -= 1
            left += 1
        if right - left > max_length:
            max_length = right - left
    return max_length


def lengthOfLongestSubstring(s: str) -> int:
    begin = res = 0
    hashTable = dict()
    # hashFun : char : index
    for idx, c in enumerate(s):
        if c in hashTable and hashTable[c] >= begin:
            # 记录每个元素的位置，如果有重复元素就把begin移动到重复元素的后一个位置，为了计算idx-begin长度
            begin, res = hashTable[c] + 1, max(res, idx - begin)
        hashTable[c] = idx

    return max(res, len(s) - begin)


res = lengthOfLongestSubstring(s)
print()

s = "cbaebabacd"
p = "abc"
s = "abab"
p = "ab"


# https://leetcode.cn/problems/find-all-anagrams-in-a-string/
def findAnagrams(s: str, p: str) -> List[int]:
    n = len(p)
    res = []
    windows, needs = Counter(), Counter(p)
    for i in range(len(s)):
        if i + n - 1 < len(s):
            windows = Counter(s[i:i + n])
            if windows == needs:
                res.append(i)
    return res


findAnagrams(s, p)


"""
滑动窗口左移动 右移动
https://leetcode.cn/problems/minimum-size-subarray-sum/solution/chang-du-zui-xiao-de-zi-shu-zu-by-leetcode-solutio/
O(n)
长度最小子数组
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
"""
def minSubArrayLen(self, s: int, nums: List[int]) -> int:
    if not nums:
        return 0

    n = len(nums)
    ans = n + 1
    start, end = 0, 0
    total = 0
    while end < n:
        total += nums[end]
        while total >= s:
            ans = min(ans, end - start + 1)
            total -= nums[start]
            start += 1
        end += 1

    return 0 if ans == n + 1 else ans


"""
void slidingWindow(string s) {
    unordered_map<char, int> window;
    
    int left = 0, right = 0;
    while (right < s.size()) {
        // c 是将移入窗口的字符
        char c = s[right];
        // 增大窗口
        right++;
        // 进行窗口内数据的一系列更新
        ...

        /*** debug 输出的位置 ***/
        printf("window: [%d, %d)\n", left, right);
        /********************/
        
        // 判断左侧窗口是否要收缩
        while (window needs shrink) {
            // d 是将移出窗口的字符
            char d = s[left];
            // 缩小窗口
            left++;
            // 进行窗口内数据的一系列更新
            ...
        }
    }
}
"""
