class Solution:
    def plusOne(self, digits):
        num = 0
        for digit in digits:
            num *=10
            num += digit
        num +=1
        num = str(num)
        return_num = []
        print(num)
        for sub_num in num:
            return_num.append(int(sub_num))
        return return_num


digits = [9,9,9]
solution = Solution()
num = solution.plusOne(digits)
print(num)