# 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
# 贪心
# class Solution:
#     def maxSubArray(self, nums):
#         if nums == None:
#             return None
#         ans = nums[0]
#         sum = 0
#         for num in nums:
#             if sum  <0:
#                 sum = 0
#             sum += num
#             if sum > ans:
#                 ans = sum
#         return ans

# dp
class Solution:
    def maxSubArray(self, nums):
        if nums == None:
            return None
        ans = float("-inf")
        sum = 0
        for num in nums:
            if sum < 0:
                sum = 0
            sum += num
            if sum > ans:
                ans = sum
        return ans

# list = [-2,1,-3,4,-1,2,1,-5,4]
# list = [-1,-2,-3]
list = [-3,-2,-1]

solution = Solution()
max_sum = solution.maxSubArray(list)
print("max_sum",max_sum)


