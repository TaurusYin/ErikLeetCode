from collections import defaultdict

"""
https://leetcode.cn/problems/24-game/
"""
class Solution:
    def judgePoint24(self, cards) -> bool:
        error = 1e-8

        dct = defaultdict(list)
        # 枚举一个数的算术结果
        for i in range(4):
            dct[1].append([[i], cards[i]])
        # 枚举两个数的算术结果
        for i in range(4):
            for j in range(i + 1, 4):
                dct[2].append([[i, j], cards[i] + cards[j]])
                dct[2].append([[i, j], cards[i] - cards[j]])
                dct[2].append([[i, j], -cards[i] + cards[j]])
                dct[2].append([[i, j], cards[i] * cards[j]])
                dct[2].append([[i, j], cards[i] / cards[j]])
                dct[2].append([[i, j], cards[j] / cards[i]])
        # 枚举三个数的算术结果
        for i in range(4):
            for lst, value in dct[2]:
                if i not in lst:
                    dct[3].append([lst + [i], cards[i] + value])
                    dct[3].append([lst + [i], cards[i] - value])
                    dct[3].append([lst + [i], -cards[i] + value])
                    dct[3].append([lst + [i], cards[i] * value])
                    dct[3].append([lst + [i], value / cards[i]])
                    if value:
                        dct[3].append([lst + [i], cards[i] / value])

        # 检查能否得出目标值注意浮点数精度到达一定程度就行
        def check(value1, value2):
            if abs(abs(value1 + value2) - 24) < error:
                return True
            if abs(abs(value1 - value2) - 24) < error:
                return True
            if abs(abs(value1 * value2) - 24) < error:
                return True
            if value2 and abs(abs(value1 / value2) - 24) < error:
                return True
            if value1 and abs(abs(value2 / value1) - 24) < error:
                return True
            return False

        # 两两组合
        for lst1, value1 in dct[2]:
            for lst2, value2 in dct[2]:
                if len(set(lst1 + lst2)) == 4 and check(value1, value2):
                    return True
        # 三一组合
        for lst1, value1 in dct[1]:
            for lst2, value2 in dct[3]:
                if len(set(lst1 + lst2)) == 4 and check(value1, value2):
                    return True
        return False
