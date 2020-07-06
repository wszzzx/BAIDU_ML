class Solution:
    def moveZeroes(self, nums):
        zero_count = 0
        nums_len = len(nums)
        for i in range(nums_len):
            num = nums[i]
            if num ==0:
                zero_count +=1
            else:
                if zero_count>0:
                    nums[i-zero_count] = num
        for i in range(nums_len -zero_count,nums_len):
            nums[i] = 0



nums = [0,0,1]
Solution().moveZeroes(nums)
print("nums",nums)