# class Solution:
#     def maxProfit(self, prices):
#         len_price = len(prices)
#         if len_price==0:
#             return None
#         index = 0
#         max_profit = 0
#         while(index <len_price-1):
#             next_index = index +1
#             buy_price = prices[index]
#             while(next_index < len_price):
#                 sell_price = prices[next_index]
#                 if sell_price >= buy_price and sell_price - buy_price >max_profit:
#                     max_profit = sell_price - buy_price
#                 next_index +=1
#             index +=1
#         return max_profit
# class Solution:
#     def maxProfit(self, prices):
#         max_profit = 0
#         for i in range(len(prices)):
#             for j in range(i+1,len(prices)):
#                 max_profit = max(max_profit,prices[j]- prices[i])
#         return max_profit

class Solution:
    def maxProfit(self, prices):
        max_profit = 0
        min_price = float('inf')
        for price in prices:
            min_price = min(min_price,price)
            max_profit = max(max_profit,price - min_price)
        return max_profit
solution = Solution()
list = [1,6,2,3,8]
max_profit = solution.maxProfit(list)
print("max_profit",max_profit)

