class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        len1 = len(nums1)
        len2 = len(nums2)
        if len1 < len2:
            list1 = nums1
            list2 = nums2
        else:
            list1 = nums2
            list2 = nums1
        m = int(nums1/2)
        while True :
            if (len1 + len2)%2 == 0:
                n = (len1 + len2) - m
            else:
                n = (len1 + len2 + 1)/2 - m
            

