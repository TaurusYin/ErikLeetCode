"""

示例 1:

输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/open-the-lock
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""

def openLock(self, deadends: List[str], target: str) -> int:
    forbid = set()
    for dead in deadends:
        forbid.add(tuple([int(d) for d in dead]))

    target = [int(d) for d in target]
    stack = [[0, 0, 0, 0]] if (0, 0, 0, 0) not in forbid else None
    forbid.add((0, 0, 0, 0))
    step = 0
    while stack:
        nex = []
        for state in stack:
            if state == target:
                return step
            for i in range(4):
                for j in [-1, 1]:
                    cur = state[:]
                    cur[i] = (cur[i] + j) % 10
                    if tuple(cur) not in forbid:
                        forbid.add(tuple(cur))
                        nex.append(cur)
        stack = nex
        step += 1
    return -1


