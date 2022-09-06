from typing import List



# https://leetcode.cn/problems/n-queens/solution/dai-ma-sui-xiang-lu-51-n-queenshui-su-fa-2k32/
def solveNQueens(self, n: int) -> List[List[str]]:
    if not n: return []
    board = [['.'] * n for _ in range(n)]
    res = []

    def isVaild(board, row, col):
        # 判断同一列是否冲突
        for i in range(len(board)):
            if board[i][col] == 'Q':
                return False
        # 判断左上角是否冲突
        i = row - 1
        j = col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        # 判断右上角是否冲突
        i = row - 1
        j = col + 1
        while i >= 0 and j < len(board):
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        return True

    def backtracking(board, row, n):
        # 如果走到最后一行，说明已经找到一个解
        if row == n:
            temp_res = []
            for temp in board:
                temp_str = "".join(temp)
                temp_res.append(temp_str)
            res.append(temp_res)
        for col in range(n):
            if not isVaild(board, row, col):
                continue
            board[row][col] = 'Q'
            backtracking(board, row + 1, n)
            board[row][col] = '.'

    backtracking(board, 0, n)
    return res

