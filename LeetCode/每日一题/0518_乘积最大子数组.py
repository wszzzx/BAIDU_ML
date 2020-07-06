class Solution:
    def maxProduct(self, nums):
        # [-4, -3, -2]
        min_value = nums[0]
        max_value = nums[0]
        ans = nums[0]
        for i in range(1,len(nums)):
            min_temp = min_value
            max_temp = max_value
            max_value = max(max_temp*nums[i],max(min_temp*nums[i],nums[i]))
            min_value = min(min_temp*nums[i],min(max_temp*nums[i],nums[i]))
            ans = max(max_value,ans)
            # print(min_value,max_value,ans)
        return ans

    # def maxProduct(self, nums):
    #     min_value = 1
    #     max_value = 1
    #     ans = float('-inf')
    #     for num in nums:
    #         if num ==0 :
    #             min_value = 0
    #             max_value = 0
    #         else:
    #             if min_value ==0:
    #                 temp = num
    #             else:
    #                 temp *=num
    #         ans = max(ans,temp,num)
    #     return ans


list = [-4,-3,-2]
ans = Solution().maxProduct(list)
print("ans",ans)