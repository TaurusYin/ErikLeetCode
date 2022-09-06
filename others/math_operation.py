from bisect import bisect_left
from typing import List

"""
https://leetcode.cn/problems/single-number/
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
"""


def singleNumber(self, nums: List[int]) -> int:
    res = 0
    for s in nums:
        res ^= s
    return res


"""
https://leetcode.cn/problems/single-number-ii/solution/by-liupengsay-l0y4/
给你一个整数数组 nums ，除某个元素仅出现 一次 外，其余每个元素都恰出现 三次 。请你找出并返回那个只出现了一次的元素。
输入：nums = [2,2,3,2]
输出：3
"""


def singleNumber(self, nums: List[int]) -> int:
    nums.sort()
    i = 0
    n = len(nums)
    while i < n:
        if i + 2 < n:
            if nums[i + 2] != nums[i]:
                return nums[i]
            else:
                i += 3
            continue
        return nums[i]


"""
给定一个整数数组 nums，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。你可以按 任意顺序 返回答案。
https://leetcode.cn/problems/single-number-iii/
"""
def singleNumber(self, nums: List[int]) -> List[int]:
    s = set()
    for num in nums:
        if num in s:
            s.remove(num)
        else:
            s.add(num)
    return list(s)


"""
给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。
https://leetcode.cn/problems/missing-number/
"""


def missingNumber(self, nums: List[int]) -> int:
    res = 0
    res ^= len(nums)
    for idx, x in enumerate(nums):
        res ^= x ^ idx
        print(res)
    return res


def _missingNumber(self, nums: List[int]) -> int:
    n = len(nums)
    expect = (0 + n) * (n + 1) / 2
    sum = 0
    for x in nums:
        sum += x
    return int(expect - sum)


"""
编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。
输入：00000000000000000000000000001011
输出：3
https://leetcode.cn/problems/number-of-1-bits/
"""


def hammingWeight(self, n: int) -> int:
    ans = 0
    while n:
        n &= (n - 1)  # n & (n-1) 去掉后面的1 O(logN)
        ans += 1
    return ans


"""
判断一个数是不是 2 的指数
2^0 = 1 = 0b0001
2^1 = 2 = 0b0010
2^2 = 4 = 0b0100
一个数如果是 2 的指数，那么它的二进制表示一定只含有一个 1：2的幂次方的二进制有一个特性，那就是他的二进制只有一个1，并且都是和他的n-1的0,1位置互相取反，因此我们可以使用n和n-1的二进制进行按照位与的操作，我们就可以得到结果为0，因此得出结果。
https://leetcode.cn/problems/power-of-two/submissions/
"""


def isPowerOfTwo(self, n: int) -> bool:
    if n <= 0:
        return False
    return n & (n - 1) == 0


"""
[1,n] 中质因子 pp 的个数为
给定一个整数 n ，返回 n! 结果中尾随零的数量。 O(logN)
提示 n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1
https://leetcode.cn/problems/factorial-trailing-zeroes/solution/jie-cheng-hou-de-ling-by-leetcode-soluti-1egk/
"""


def trailingZeroes(self, n: int) -> int:
    ans = 0
    while n:
        ans += n // 5
        n //= 5
    return ans


"""
https://leetcode.cn/problems/preimage-size-of-factorial-zeroes-function/
阶乘函数后 K 个零
# 时间复杂度在 O(logk)，在[4k,5k]上二分查找 等比和 为 k的x
"""


def preimageSizeFZF(self, k: int) -> int:
    def zeta(n: int) -> int:
        res = 0
        while n:
            n //= 5
            res += n
        return res

    def nx(k: int) -> int:
        return bisect_left(range(5 * k), k, key=zeta)  # key 比较键

    return nx(k + 1) - nx(k)


"""
https://leetcode.cn/problems/count-primes/solution/shi-yong-lie-biao-ji-lu-zhi-shu-de-ge-sh-59nh/
质数个数，
"""


def countPrimes(self, n: int) -> int:
    # 没有小于2的质数，直接返回0
    if n <= 2:
        return 0
    else:
        # 0、1不是质数，直接标记为False，先假设2后面的全部是质数，先赋True
        l = [False] * 2 + [True] * (n - 2)

    for i in range(2, n):
        # 一开始出现的i都为质数，把i的倍数全部设为False
        if l[i] == True:
            for j in range(2 * i, n, i):
                l[j] = False
    # 返回列表种True的个数
    return l.count(True)


"""
https://leetcode.cn/problems/super-pow/
https://leetcode.cn/problems/super-pow/solution/pythonyu-yan-tong-yong-jie-fa-yuan-li-ji-4dgj/

Your task is to calculate ab mod 1337 where a is a positive integer and b is an extremely large positive integer given in the form of an array.
快速幂思想在于把指数变小，当指数为11时，常规做法需要11次循环。而使用快速幂可以降低循环次数，原理在于对指数的分解
只需要对指数11进行右移处理3次,每次判断改二进制位是否为1
"""


def superPow(self, a: int, b: List[int]) -> int:
    def quick_pow(num1, num2):
        # 题目num可能为2^31 - 1,第一次num * num后会溢出,先对num取余
        num1 = num1 % MOD
        ret = 1
        while num2:
            # 判断该位是否为1,是的话累计入答案
            if num2 & 1:
                # 累计答案,取余
                ret = ret * num1 % MOD
            # 每次指数右移一位底数需要累积并取余
            num1 = num1 * num1 % MOD
            num2 >>= 1
        return ret

    n = len(b)
    ans = 1
    MOD = 1337
    for i in range(n - 1, -1, -1):
        # 把数组中的数作为指数再和处理过的底数做运算
        ans = ans * quick_pow(a, b[i]) % MOD
        # 处理底数
        a = quick_pow(a, 10)

    return ans
