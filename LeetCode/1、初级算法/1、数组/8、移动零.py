class Solution:
    def moveZeroes(self, nums):
        index = 0
        for num in nums:
            if num !=0:
                nums[index] = num
                index +=1
        print("index",index)
        for i in range(index,len(nums)):
            nums[i] = 0
solution = Solution()
nums = [0,1,0,2,0,3]
solution.moveZeroes(nums)
print("nums",nums)



                
