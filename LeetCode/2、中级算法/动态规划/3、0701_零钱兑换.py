class Solution:
    def coinChange(self, coins, amount: int) -> int:
        coins_len = len(coins)
        if coins_len == 0:
            return -1
        if amount == 0:
            return 0
        amount_list = [-1 for i in range(amount+1)]
        amount_list[0] = 0
        for i in range(1,amount+1):
            min_value = float('inf')
            for coin in coins:
                if i - coin >=0 and amount_list[i-coin] !=-1:
                    min_value_temp = amount_list[i-coin] +1
                    min_value = min(min_value,min_value_temp)
            if min_value < float('inf'):
                amount_list[i] = min_value
        # print(amount_list)
        return amount_list[amount]


coins = [2]
amount = 3
minvalue = Solution().coinChange(coins,amount)
print(minvalue)




