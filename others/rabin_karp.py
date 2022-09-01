from typing import List


def search(a, b):
    """
    判断a是不是b的子串
    """
    hashA, hashB = 0, 0
    aN = [ord(c)-ord('A')+1 for c in a]
    bN = [ord(c)-ord('A')+1 for c in b]
    m, n = len(aN), len(bN)
    for i in range(m):
        hashA = 26*hashA + aN[i]
        hashB = 26*hashB + bN[i]
    for j in range(m-1, n):
        if j > m-1:
            hashB -= bN[j-m]*(26**(m-1))
            hashB = hashB*26 + bN[j]
        if hashB == hashA:
            if a == b[j-m+1:j+1]:
                return True
    return False
search( "AABA", "ACAADAABAAAAABABAA")


def cal(ops, R, L, val, number):
    if ops == 'add_last':
        return number * R + val
    if ops == 'remove_first':
        return number - val * (R ** (L - 1))

# https://labuladong.github.io/algo/2/20/28/
def findRepeatedDnaSequences(s: str) -> List[str]:
    nums = len(s) * [0]
    R = 10; L = 10
    for i in range(len(s)):
        if s[i] == 'A': nums[i] = 0
        if s[i] == 'G': nums[i] = 1
        if s[i] == 'C': nums[i] = 2
        if s[i] == 'T': nums[i] = 3
    seen = {} ; res = []
    left = right = 0
    number = 0
    while right < len(s):
        right_elem = nums[right]
        right += 1
        number = cal(ops='add_last', R=R, L=L, val=right_elem, number=number)
        if right - left == L:
            left_elem = nums[left]
            if number in seen.keys():
                res.append(s[left:right])
            else:
                seen[number] = True
            number = cal(ops='remove_first', R=R, L=L, val=left_elem, number=number)
            left += 1
    return list(set(res))

s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
# s = "AAAAAAAAAAAAA"
x = findRepeatedDnaSequences(s)
print()



