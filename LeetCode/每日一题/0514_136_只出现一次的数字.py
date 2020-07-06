class Solution:
    def singleNumber(self, nums):
        ans = 0
        for num in nums:
            ans^=num
        return ans
solution = Solution()
nums = [1,2,3,1,2]
ans = solution.singleNumber(nums)
print("ans",ans)