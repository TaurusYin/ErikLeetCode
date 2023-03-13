def combine(n: int, k: int):
    result = []
    path = []
    def backtracking(n, k, start_index):
        if len(path) == k:
            result.append(path[:])
            print('result:{}'.format(result))
            return

        end_index = n + 1 - (k - len(path))  # (k - len(path)) 还需要取多少个数
        for i in range(start_index, end_index + 1):  # cut from n+1 -> n - (k - len(path)) + 2
            print('path:{}'.format(path))
            path.append(input[i-1])
            backtracking(n, k, i + 1)
            path.pop()

    backtracking(n, k, 1)
    return result

input = ['A', 'B', 'C', 'D']
combine(4,3)
