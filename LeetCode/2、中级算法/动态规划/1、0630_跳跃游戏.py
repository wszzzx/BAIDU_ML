class Solution:
    def canJump(self, nums) -> bool:
        length = len(nums)
        max_length = 0
        for i ,num in enumerate(nums):
            if i <= max_length:
                max_length = max(i+num,max_length)
                if max_length >= length -1:
                    return True
            else:
                return False
        # max_length = 0
        # for i,num in enumerate(nums):
        #     if i>max_length:
        #         return False
        #     max_length = max(max_length,i+num)
        # return True



# nums = [2,3,1,1,4]
nums = [3,2,1,0,4]
flag = Solution().canJump(nums)
print("flag",flag)