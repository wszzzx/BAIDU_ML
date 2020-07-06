class Solution:
    def removeDuplicates(self, nums):
        if len(nums) <=1:
            return len(nums)
        index = 0
        for i in range(1,len(nums)):
            if nums[i] != nums[index]:
                if i != index+1:
                    nums[index+1] = nums[i]
                index +=1
        return index +1

solution = Solution()
nums = [0,0,0,1,1,1,2,2,2,2,2]
length = solution.removeDuplicates(nums)
print(length,nums[:length])





