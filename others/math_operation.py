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
https://leetcode.cn/problems/first-missing-positive/
输入：nums = [1,2,0]
输出：3
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数
O(n)
其实负数和大于N的数可以忽略，都视为空座位。减少原先为了方便理解而做的操作。
还有一个坑点是有两个相同的正票（1~N的整数），只有1个人能对号入座，无论谁入座都可以，要避免在两个相同正票互换座位中死循环！所以应以“先到先得”为原则，已经对号入座者不变！
代码中判断没有对号入座的a!=nums[a-1]不能替换为a!=i+1(等价于nums[i]!=i+1)。否则当nums[i]==a==nums[a-1]时（有两个相同正票）会死循环。

链接：https://leetcode.cn/problems/first-missing-positive/solution/zhao-zuo-wei-de-gu-shi-onyuan-di-jie-fa-by-java_le/
"""


def firstMissingPositive(self, nums: List[int]) -> int:
    for a in nums:  # 遍历每个座位，记当前坐着a号乘客
        while 0 < a <= len(nums) and a != nums[a - 1]:  # 乘客a是正票但坐错了! 其座位被 ta=nums[a-1]占了
            nums[a - 1], a = a, nums[a - 1]  # a和ta两人互换则a对号入座。此后ta相当于新的a，去找自己的座位（循环执行）
    for i in range(len(nums)):
        if i + 1 != nums[i]: return i + 1  # 找到首个没有对号入座的nums[i]!=i+1
    return len(nums) + 1  # 满座，返回N+1


"""
给你一个含 n 个整数的数组 nums ，其中 nums[i] 在区间 [1, n] 内。请你找出所有在 [1, n] 范围内但没有出现在 nums 中的数字，并以数组的形式返回结果。
输入：nums = [4,3,2,7,8,2,3,1]
输出：[5,6] O(N)
https://leetcode.cn/problems/find-all-numbers-disappeared-in-an-array/solution/tong-pai-xu-ji-yu-yi-huo-yun-suan-jiao-huan-liang-/
"""
def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
    # 基于异或运算交换数组两个位置的元素，不使用额外的空间
    def swap(nums, index1, index2):
        # 这一步是必要的，否则会使得一个数变成 0
        if index1 == index2:
            return
        nums[index1] = nums[index1] ^ nums[index2]
        nums[index2] = nums[index1] ^ nums[index2]
        nums[index1] = nums[index1] ^ nums[index2]

    for i in range(len(nums)):
        while nums[i] != nums[nums[i] - 1]:
            # 如果不在位置上，并且它将要去的那个位置上的数不等于自己，则交换
            swap(nums, i, nums[i] - 1)

    return [i + 1 for i, num in enumerate(nums) if num != i + 1]


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
https://leetcode.cn/problems/powx-n/solution/xiong-mao-shua-ti-python3-di-gui-by-lotuspanda/
输入：x = 2.10000, n = 3
输出：9.26100
"""


class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if n == 1:
            return x
        if n < 0:
            n = -n
            half = self.myPow(x, n // 2)  # 注意，这里用地板除法
            if n % 2 == 0:
                return 1 / (half * half)  # 这里注意用算数的除法
            else:
                return 1 / (half * half * x)
        else:
            half = self.myPow(x, n // 2)
            if n % 2 == 0:
                return half * half
            else:
                return half * half * x


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


"""
https://leetcode.cn/problems/sqrtx/solution/x-de-ping-fang-gen-by-leetcode-solution/
"""


def mySqrt(self, x: int) -> int:
    l, r, ans = 0, x, -1
    while l <= r:
        mid = (l + r) // 2
        if mid * mid <= x:
            ans = mid
            l = mid + 1
        else:
            r = mid - 1
    return ans


"""
输入：n = 2
输出：[0,1,3,2]
解释：
[0,1,3,2] 的二进制表示是 [00,01,11,10] 。
- 00 和 01 有一位不同
- 01 和 11 有一位不同
- 11 和 10 有一位不同
- 10 和 00 有一位不同
[0,2,3,1] 也是一个有效的格雷码序列，其二进制表示是 [00,10,11,01] 。
- 00 和 10 有一位不同
- 10 和 11 有一位不同
- 11 和 01 有一位不同
- 01 和 00 有一位不同

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/gray-code
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


def grayCode(self, n: int) -> List[int]:
    ans = [0] * (1 << n)
    for i in range(1 << n):
        ans[i] = (i >> 1) ^ i
    return ans


"""
给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以 字符串形式返回小数 。
如果小数部分为循环小数，则将循环的部分括在括号内
输入：numerator = 1, denominator = 2
输出："0.5"
输入：numerator = 4, denominator = 333
输出："0.(012)"
链接：https://leetcode.cn/problems/fraction-to-recurring-decimal/solution/by-isuxiz-59ks/
时空复杂度
时空复杂度均为O(L)，其中L为答案字符串的长度
其实这里应该有一个公式能把L用输入的两数的值M，N来表示的，但是我没想出来 ORZ
"""


def fractionToDecimal(self, numerator: int, denominator: int) -> str:
    # 预处理
    is_negative = numerator * denominator < 0
    numerator = abs(numerator)
    denominator = abs(denominator)

    # 得到整数商
    integer_quotient = str(numerator // denominator)
    numerator %= denominator
    numerator *= 10

    # 求小数部分
    ans = deque()
    # 存第一次见到这个要处理的数的轮次
    # 即对应的那位答案在ans中的下标
    seen_numbers = {}
    i = 0
    while numerator:
        # 如果当前要处理的数出现过
        # 说明现在求的这一位是下一个循环节的开头，没有必要求了
        if numerator in seen_numbers:
            break
        seen_numbers[numerator] = i
        ans.append(str(numerator // denominator))
        numerator %= denominator
        numerator *= 10
        i += 1

    # 如果结束时余数非零
    # 说明是检测到存在循环节才退出循环的
    # 那么添加括号
    if numerator:
        ans.insert(seen_numbers[numerator], '(')
        ans.append(')')

    # 如果小数部分有结果，那么添加小数点
    if ans:
        ans.appendleft('.')

    # 添加整数部分
    ans.appendleft(integer_quotient)

    # 添加符号
    if is_negative:
        ans.appendleft('-')

    return "".join(ans)
