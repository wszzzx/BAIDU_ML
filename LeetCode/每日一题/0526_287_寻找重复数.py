class Solution:
    def findDuplicate(self, nums):
        left = 1
        right = len(nums) -1
        ans = -1
        while(left<=right):
            mid = int((left + right) / 2)
            print("left",left,right,mid)

            count = 0
            for num in nums:
                if num <=mid:
                    count+=1
            print("count",count)
            if count > mid :
                right = mid -1
                ans = mid
            else:
                left = mid+1

        return ans
list = [1,3,4,2,2]
ans = Solution().findDuplicate(list)
print("ans",ans)


