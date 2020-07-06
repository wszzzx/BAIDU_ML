class Solution:
    def rob(self, nums) -> int:
        nums_length = len(nums)
        if nums_length ==0:
            return None
        elif nums_length ==1:
            return nums[0]
        elif nums_length == 2:
            return max(nums[0],nums[1])
        else:
            ans_list = [nums[0],max(nums[0],nums[1])]
            for i in range(2,nums_length):
                ans_list.append(max(ans_list[i-1],ans_list[i-2]+nums[i]))
            return ans_list[-1]
nums =[7,1,3,11]
ans = Solution().rob(nums)
print("ans",ans)
