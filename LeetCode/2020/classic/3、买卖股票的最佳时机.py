class Solution:
    def maxProfit(self, prices):
        if len(prices) <= 1:
            return 0
        buy_price = prices[0]
        sell_price = prices[0]
        max_profoib = 0
        for price in prices:
            if price > sell_price:
                sell_price = prices
                profoib = price - buy_price
                if profoib >max_profoib:
                    max_profoib = profoib
            if price < buy_price:
                buy_price = price
                sell_price = price
        return max_profoib

list = [2,1,3,4]


