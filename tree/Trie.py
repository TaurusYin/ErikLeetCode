import collections
import heapq
from collections import defaultdict


class Node(object):
    def __init__(self, val: int):
        self.val = val

    def __repr__(self):
        return f'Node value: {self.val}'

    def __lt__(self, other):
        return self.val < other.val

    def __eq__(self, other):
        return self.val == other.val

    def __gt__(self, other):
        return self.val > other.val


heap = [Node(2), Node(0), Node(1), Node(4), Node(2)]
heap = [2, 0, 1, 4, 2]
heapq.heapify(heap)
print(heap)  # output: [Node value: 0, Node value: 2, Node value: 1, Node value: 4, Node value: 2]

heapq.heappop(heap)
print(heap)  # output: [

"""
https://leetcode.cn/problems/implement-trie-prefix-tree/

"""
class Trie(object):

    def __init__(self):
        #collections.defaultdict(dict)当key为空时，返回dict的默认值{}
        self.child = collections.defaultdict(dict)


    def insert(self, word):
        """
        :type word: str
        :rtype: None
        """
        node = self.child
        for s in word:
            if s not in node:
                node[s] = collections.defaultdict(dict)
        #遍历下一个节点
            node = node[s]

        #在字符串的末尾添加一个‘#’，表示结束标志
        node['#'] = {}


    def search(self, word):
        """
        :type word: str
        :rtype: bool
        """
        node = self.child
        for s in word:
           #if s not in node.keys() : 此处为keys()而非key()
            if s not in node: #判断当前字符是否在字典的key值内
                return False
            node = node[s]
        return '#' in node

    def startsWith(self, prefix):
        """
        :type prefix: str
        :rtype: bool
        """
        node = self.child
        for s in prefix:
            if s not in node:
                return False
            node = node[s]
        return True


trie = Trie()
trie.insert("apple")
x = trie.search("apple")
y = trie.search("app")
z = trie.startsWith("app")
trie.insert("app")
s = trie.search("app")



class TrieII:

    def __init__(self):
        self.node = defaultdict(dict)

    def insert(self, word: str) -> None:
        cur = self.node[word[0]]
        for n in word[1:]:
            cur['prefix'] = cur['prefix'] + 1 if 'prefix' in cur else 1
            if n in cur:
                cur = cur[n]
            else:
                cur[n] = {}
                cur = cur[n]
        cur['end'] = cur['end'] + 1 if 'end' in cur else 1
        cur['prefix'] = cur['prefix'] + 1 if 'prefix' in cur else 1

    def countWordsEqualTo(self, word: str) -> int:
        cur = self.node
        for n in word:
            if n in cur:
                cur = cur[n]
            else:
                return 0
        return cur['end'] if 'end' in cur else 0

    def countWordsStartingWith(self, prefix: str) -> int:
        cur = self.node
        for n in prefix:
            if n in cur:
                cur = cur[n]
            else:
                return 0
        return cur['prefix']

    def erase(self, word: str) -> None:
        cur = self.node
        for n in word:
            cur = cur[n]
            if 'prefix' in cur:
                cur['prefix'] -= 1
        cur['end'] -= 1



"""
https://leetcode.cn/problems/longest-word-in-dictionary/solution/ci-dian-zhong-zui-chang-de-dan-ci-by-lee-k5gj/
O(sum(Li单词长度))
O(sum(Li单词长度))
输入：words = ["w","wo","wor","worl", "world"]
输出："world"
解释： 单词"world"可由"w", "wo", "wor", 和 "worl"逐步添加一个字母组成。
"""
def longestWord(self, words: List[str]) -> str:
    t = Trie()
    for word in words:
        t.insert(word)
    longest = ""
    for word in words:
        if t.search(word) and (len(word) > len(longest) or len(word) == len(longest) and word < longest):
            longest = word
    return longest


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



#Definition of TrieNode:
class TrieNode:
    def __init__(self):
        # <key, value>: <Character, TrieNode>
        self.children = collections.OrderedDict()


class Serializetion:
    '''
    @param root: An object of TrieNode, denote the root of the trie.
    This method will be invoked first, you should design your own algorithm
    to serialize a trie which denote by a root node to a string which
    can be easily deserialized by your own "deserialize" method later.
    '''

    def serialize(self, root):
        # Write your code here
        if root is None:
            return ""

        data = ""
        for key, value in root.children.items():
            data += key + self.serialize(value)

        return "<%s>" % data

    '''
    @param data: A string serialized by your serialize method.
    This method will be invoked second, the argument data is what exactly
    you serialized at method "serialize", that means the data is not given by
    system, it's given by your own serialize method. So the format of data is
    designed by yourself, and deserialize it here as you serialize it in 
    "serialize" method.
    '''

    def deserialize(self, data):
        # Write your code here
        if data is None or len(data) == 0:
            return None

        root = TrieNode()
        current = root
        path = []
        for c in data:
            if c == '<':
                path.append(current)
            elif c == '>':
                path.pop()
            else:
                current = TrieNode()
                if len(path) == 0:
                    print(c, path)
                path[-1].children[c] = current

        return root
