class Solution:
    def isValidSudoku(self, board) -> bool:
        lists = [[[] for i in range(3)] for i in range(3)]
        row_lists = [[] for i in range(9)]
        col_lists = [[] for i in range(9)]
        for i in range(len(board)):
            for j in range(len(board[0])):
                num = board[i][j]
                if num!=".":
                    if num not in row_lists[i]:
                        row_lists[i].append(num)
                    else:
                        return False
                    if num not in col_lists[j]:
                        col_lists[j].append(num)
                    else:
                        return False
                    if num not in lists[int(i/3)][int(j/3)]:
                        lists[int(i / 3)][int(j / 3)].append(num)
                    else:
                        return False
        return True

lists =[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
flag = Solution().isValidSudoku(lists)
print(flag)

