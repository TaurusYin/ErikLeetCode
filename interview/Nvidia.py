"""
a[M,N]
A[M,N]


A[p,q] = Sum(a[i,j], 0<=i<=p, 0<=j<=q)
1) Given A[p-1, q-1], A[p-1, q], A[p, q-1], a[p,q], How to calculate A[p,q]
A[p,q]  = A[p-1, q] + A[p, q-1] - A[p-1,q-1] + a[p, q]

2) Given a[M,N], calculate A[M,N]

"""
from typing import List


def matrixSum(a: List[List[int]]) -> List[List[int]]:
    m, n = len(a), len(a[0])
    pre_sum = [[0] * (1 + n) for _ in range(1 + m)]
    pre_sum = a
    for i in range(m - 1):
        for j in range(n - 1):
            pre_sum[i + 1][j + 1] = pre_sum[i + 1][j] + pre_sum[i][j + 1] - pre_sum[i][j] + a[i][j]
    return pre_sum


# Intersaction Over Union
# Box1 (x1,y1) (x2,y2)
# Box2 (x3,y3) (x4,y4)


overlap_width = min(x2, x4) - max(x1, x3)
overlap_height = min(y2, y4) - max(y1, y3)
overlap = max(overlap_width, 0) * max(overlap_height, 0)

# n = 1
# n = 2 , res = 1
# n = 3 , (a,b,c)  1

# n = 4  (a,b,c,d)   (a,b) , (c,d)
# (a,b)


# list longest increase sequence
input = [13, 1, 6, 7, 8, 5, 1, 2, 3, 4, 5, 6]


def find_longest(alist):
    n = 1
    res = 0
    seq = []
    seq_result = []
    for i in range(1, len(alist)):
        if alist[i] > alist[i - 1]:
            n += 1
            # res = max(res, n)
            seq.append(alist[i])
        else:
            if res < n:
                res = n
            n = 1
            seq_result = seq
            seq = []
            seq.append(alist[i])

        if i == len(alist) - 1:
            if res < n:
                res = n
            n = 1
            seq_result = seq
    return res, seq_result


ans_len, seq = find_longest(input)
print(ans_len, seq)
