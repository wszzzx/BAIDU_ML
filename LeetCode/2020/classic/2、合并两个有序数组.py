class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i = m-1
        j = n-1
        index = m+n-1
        while(i>=0 and j >=0):
            if nums1[i] >nums2[j]:
                nums1[index] = nums1[i]
                i-=1
                index -=1
            else:
                nums1[index] = nums2[j]
                j-=1
                index -=1

        if j >=0:
            nums1[:j+1] = nums2[:j+1]

