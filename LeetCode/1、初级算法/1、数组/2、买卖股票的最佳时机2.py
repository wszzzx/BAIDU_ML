class Solution:
    def maxProfit(self, prices):
        if len(prices) <=1:
            return 0
        max_Profit = 0
        min_num = prices[0]
        for i in range(1,len(prices)):
            if prices[i] > min_num:
                max_Profit +=prices[i] - min_num
                min_num = prices[i]
            else:
                min_num =  prices[i]
        return max_Profit
prices = [1]
solution = Solution()
maxProfit = solution.maxProfit(prices)
print("maxProfit",maxProfit)

