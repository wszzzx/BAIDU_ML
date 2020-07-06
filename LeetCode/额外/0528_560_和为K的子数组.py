class Solution:
    def subarraySum(self, nums, k) -> int:
        sum_dict = {0:1}
        ans = 0
        sum = 0
        for num in nums:
            sum +=num
            if sum -k in sum_dict.keys():
                ans +=sum_dict[sum-k]
            if sum not in sum_dict.keys():
                sum_dict[sum] = 1
            else:
                sum_dict[sum] +=1
        return ans

nums = [1]
ans = Solution().subarraySum(nums,0)
print("ans",ans)

