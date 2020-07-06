# class Solution:
#     def twoSum(self,  list1, target):
#         length = len(list1)
#         dict1 = {}
#         for i in range(length):
#             dict1[list1[i]] = i
#         for num in list1:
#             num1 = target - num
#             if num1 in dict1.keys() and list1.index(num)!=dict1.get(num1):
#                 return [list1.index(num),dict1.get(num1)]
#complement
class Solution:
    def twoSum(self,  list1, target):
        length = len(list1)
        dict1 = {}
        for i in range(length):
            num1 = list1[i]
            num2 = target - num1
            if num2 not in dict1.keys():
                dict1[num1] = i
            else:
                return [dict1.get(num2),i]

list1 = [2,3,3,5]

solution = Solution()
list = solution.twoSum(list1,7)
print(list)
