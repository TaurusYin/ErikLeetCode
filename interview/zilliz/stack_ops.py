import heapq
from sortedcontainers import SortedList


class Node:
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value < other.value


class MaxStack:
    def __init__(self):
        self.idx, self.stk, self.sl = 0, dict(), SortedList()

    def push(self, x: int) -> None:
        self.stk[self.idx] = x
        self.sl.add((x, self.idx))
        self.idx += 1

    def pop(self) -> int:
        i, x = self.stk.popitem()
        self.sl.remove((x, i))
        return x

    def top(self) -> int:
        return next(reversed(self.stk.values()))

    def peekMax(self) -> int:
        return self.sl[-1][0]

    def popMax(self) -> int:
        x, i = self.sl.pop()
        self.stk.pop(i)
        return x


class MaxStack:
    # 自增序列
    # 最大堆
    # soft delete
    def __init__(self):
        self.heap = []
        self.stack = []
        self.popped_set = set()
        self.count = 0

    def push(self, x: int) -> None:
        item = (-x, -self.count)
        self.stack.append(item)
        heapq.heappush(self.heap, item)
        self.count += 1

    def pop(self) -> int:
        self.clear_pop_in_stack()  # 先清理标记
        item = self.stack.pop()  # 弹出当前元素，加入删除标记
        self.popped_set.add(item)
        return -item[0]

    def top(self) -> int:
        self.clear_pop_in_stack()  # 先清理标记
        item = self.stack[-1]
        return -item[0]

    def peekMax(self) -> int:
        self.clear_pop_in_heap()  # 先清理标记
        item = self.heap[0]
        return -item[0]

    def popMax(self) -> int:
        self.clear_pop_in_heap()
        item = heapq.heappop(self.heap)
        self.popped_set.add(item)  # 标记堆中pop的元素
        return -item[0]

    # 每次stack操作时先清空stack里标记删除的元素
    def clear_pop_in_stack(self):
        while self.stack and self.stack[-1] in self.popped_set:
            self.popped_set.remove(self.stack[-1])
            self.stack.pop()

    # 每次heap操作时先清空heap里标记删除的元素
    def clear_pop_in_heap(self):
        while self.heap and self.heap[0] in self.popped_set:
            self.popped_set.remove(self.heap[0])
            heapq.heappop(self.heap)


if __name__ == '__main__':
    st = MaxStack()
    st.push(5)
    st.push(1)
    st.push(5)
    st.popMax()
    st.peekMax()
    print()
