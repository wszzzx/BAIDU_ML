class Solution:
    # def mincostTickets(self, days, costs):
    #     periods =[1,7,30]
    #     def dp(i):
    #         if i >365:
    #             return 0
    #         elif i not in days:
    #             return dp(i+1)
    #         else:
    #             return min(dp(i + c) + d for c,d in zip(periods,costs))
    #     return dp(1)
    # def mincostTickets(self, days, costs):
    #     length = days[-1] +1
    #     dp = [0 for _ in range(length)]
    #     for i in range(1,length):
    #         if i not in days:
    #             dp[i] = dp[i-1]
    #         else:
    #             dp[i] = min(dp[max(i-1,0)]+costs[0],dp[max(i-7,0)]+costs[1],dp[max(i-30,0)]+costs[2])
    #     print("dp", dp)
    #     return dp[-1]
    def mincostTickets(self, days, costs):
        # length = len(days)
        # periods = [1,7,10]
        # dp = [0 for _ in range(length)]
        # i = length -1
        # while i >=0 :
        #     ans = float('Inf')
        #     for c,d in zip(periods,costs):
        #         j = i
        #         while j< length-1 and days[j] < days[i] + c:
        #             j+=1
        #         ans = min(ans,dp[j]+d)
        #     dp[i] = ans
        #     i -=1
        # print("dp",dp)
        # return dp[0]

        length = len(days)
        periods = [1, 7, 30]
        dp = [0 for _ in range(length)]
        for i in range(length):
            ans = float('Inf')
            for c, d in zip(periods, costs):
                j = i
                while j >=0 and days[j] + c > days[i]:
                    j -= 1
                ans = min(ans, dp[j] + d)
                # if j >=0:
                #     ans = min(ans, dp[j] + d)
                # else:
                #     ans = min(ans, 0 + d)
            dp[i] = ans
        print("dp", dp)
        return dp[-1]

solution = Solution()
# days=[1,2,3,4,6,8,9,10,13,14,16,17,19,21,24,26,27,28,29]
days=[1,2]

costs = [3,14,50]

cost = solution.mincostTickets(days,costs)
print("cost",cost)