class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0 or x ==1:
            return x
        def quickMul(n):
            ans =1
            x_contribute = x
            while(n>0):
                if n %2 ==1:
                    ans *=x_contribute
                x_contribute *= x_contribute
                n = n//2
            return ans
        return quickMul(n) if n>=0 else 1/quickMul(-n)

    # def myPow(self, x: float, n: int) -> float:
    #     def quickMul(N):
    #         if N==0:
    #             return 1.0
    #         y = quickMul(N//2)
    #         return y*y if N%2==0 else x*y*y
    #     return quickMul(n) if n>=0 else 1/quickMul(-n)
    #     if x == 0 or x ==1:
    #         return x
    #     def quick(n):
    #         if n == 0:
    #             return 1
    #         if n%2 == 0:
    #             ans = quick(n//2)
    #             return ans*ans
    #         else:
    #             ans = quick(n//2)
    #             return ans*ans*x
    #     if n <0:
    #         x = 1/x
    #         n = -n
    #     ans = quick(n)
    #     return ans

    # def myPow(self, x: float, n: int) -> float:
    #     if x == 0 or x ==1:
    #         return x
    #     if n<0:
    #         x = 1/x
    #         n = -n
    #     ans = 1
    #     for i in range(n):
    #         ans*=x
    #     return ans

solution = Solution()
ans = solution.myPow(2,4)
print(ans)