class Solution:
    def subarraysDivByK(self, nums, k) -> int:
        sum_dict = {0:1}
        ans = 0
        sum = 0
        for num in nums:
            print("sum_dict",sum_dict)
            sum +=num
            modulus = sum%k
            print("modulus",modulus)
            ans += sum_dict.get(modulus,0)
            sum_dict[modulus] =sum_dict.get(modulus,0)+1
        return ans

# class Solution:
#     def subarraysDivByK(self, A, K) -> int:
#         record = {0: 1}
#         total, ans = 0, 0
#         for elem in A:
#             total += elem
#             modulus = total % K
#             same = record.get(modulus, 0)
#             ans += same
#             record[modulus] = same + 1
#         print("record",record)
#         return ans


nums =[-1,2,9]
k = 2
ans = Solution().subarraysDivByK(nums,k)
print(ans)