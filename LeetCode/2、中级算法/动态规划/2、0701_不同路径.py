class Solution:
    def uniquePaths(self, m, n) :
        data_list = [[1 for i in range(m)] for i in range(n)]
        for i in range(1,n):
            for j in range(1,m):
                data_list[i][j] = data_list[i-1][j] + data_list[i][j-1]
        return data_list[n-1][m-1]
data_list = Solution().uniquePaths(3,7)
print("data_list",data_list)