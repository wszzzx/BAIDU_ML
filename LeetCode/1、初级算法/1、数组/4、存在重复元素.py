class Solution:
    def containsDuplicate(self, nums):
        dict = {}
        for num in nums:
            if num in dict.keys():
                return True
            else:
                dict[num] = 1
        return False
nums = [1,2,3,1]
solution = Solution()
Flag = solution.containsDuplicate(nums)
print(Flag)