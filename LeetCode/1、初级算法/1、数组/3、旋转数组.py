class Solution:
    def rotate(self, nums, k):
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums)>1:
            k = k%len(nums)
            temp_list = nums[-k:]
            nums[k:] = nums[:-k]
            nums[:k] = temp_list
nums = [1,2,3,4,5]

solution = Solution()
solution.rotate(nums,6)
print(nums)

