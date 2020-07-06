class Solution:
    def reverse(self, x):
        temp = abs(x)%2**32
        ans=0
        while(temp):
            ans= ans*10 + temp%10
            temp = int(temp/10)
        ans = ans if x>0 else -ans
        if ans in range(-2**31,2**31):
            return ans
        else:
            return 0

x = 1563847412
ans = Solution().reverse(x)
print("ans",ans,2**32)