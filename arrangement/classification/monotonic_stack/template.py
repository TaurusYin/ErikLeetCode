"""
@File    :   template.py   
@Contact :   yinjialai 
"""


def find_next_greater_elements(arr):
    stack = []
    result = {}
    for index, val in enumerate(arr):
        print(index, val)
        while stack and val > stack[-1][1]:
            stack_top_index, stack_top_val = stack.pop()
            result[stack_top_val] = val
        stack.append((index, val))
    print(result)
    return

arr = [2, 1, 5, 3, 6]
print(find_next_greater_elements(arr))  # 输出：[5, 5, 6, 6, -1]
