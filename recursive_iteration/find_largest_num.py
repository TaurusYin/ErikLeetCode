import collections
from utils.base_decorator import CommonDecorator

def find_largest_iteration(nums):
    """
    [7,1,2,11,4,5]
    stack = []
    1: 7,   stack.append(max(7, stack.pop()))
    2: 1,   stack.append(max(1, stack.pop()))
    3: 2,   stack.append(max(2, stack.pop()))
    4: 11,  stack.append(max(11, stack.pop())
    5: 5,   stack.append(max,5, stack.pop())
    :param nums:
    :return:
    """
    stack = collections.deque()
    for item in nums:
        if stack:
            stack.append(max(item, stack.pop()))
        else:
            stack.append(item)
    max_value = stack[0]
    return max_value


def find_largest_recursion(nums):
    """
    1. Call the function itself
    2. Parameters
    :param nums:
    :return:
    """
    @CommonDecorator().info()
    def find_max(nums, max_value):
        # print(locals())
        if not nums:
            return max_value
        current = nums.pop()
        if current > max_value:
            max_value = current
        res = find_max(nums, max_value)
        # print('return: {}'.format(res))
        return res

    largest_num = 0
    largest_num = find_max(nums, largest_num)
    return largest_num


if __name__ == '__main__':
    nums = [7, 1, 2, 11, 4, 5]
    stack = collections.deque()
    res = find_largest_iteration(nums)
    res = find_largest_recursion(nums)
