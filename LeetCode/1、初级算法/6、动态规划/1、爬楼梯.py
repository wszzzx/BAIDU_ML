class Solution:
    def climbStairs(self, n: int) -> int:
        ans_list = [1,2]
        if n <=2:
            return ans_list[n-1]
        for i in range(2,n):
            ans_list.append(ans_list[i-1] + ans_list[i-2])
        return ans_list[-1]
solution = Solution()
ans_list = solution.climbStairs(10)
print("ans_list",ans_list)
