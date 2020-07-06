class Solution:
    def twoSum(self, nums, target):
        nums_dict = {}
        for i in range(len(nums)):
            if target - nums[i] in nums_dict.keys():
                return[nums_dict[target - nums[i]],i]
            else:
                nums_dict[nums[i]] = i
        return []

list = [0,1,2]
target =1
return_list = Solution().twoSum(list,target)
print("return_list",return_list)