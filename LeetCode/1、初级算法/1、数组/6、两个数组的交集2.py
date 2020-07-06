class Solution:
    def intersect(self, nums1, nums2):
        dict1 = {}
        for num in nums1:
            if num not in dict1.keys():
                dict1[num] = 1
            else:
                dict1[num] +=1
        dict2 = {}
        for num in nums2:
            if num not in dict2.keys():
                dict2[num] = 1
            else:
                dict2[num] +=1
        return_list = []
        for key in dict1.keys():
            if key in dict2.keys():
                for i in range(min(dict1[key],dict2[key])):
                    return_list.append(key)
        return return_list
solution = Solution()
nums1 = [4,9,5]
nums2 = [9,4,9,8,4]
return_list =solution.intersect(nums1,nums2)
print(return_list)
