class Solution:
    def is_Palindrome(self,s):
        i = 0
        j =len(s)-1
        while(i<j):
            if s[i]!=s[j]:
                return False
            i+=1
            j-=1
        return True
    def long_str(self,i,j,s):
        if i>j:
            return ''
        elif self.is_Palindrome(s[i:j+1]):
            return s[i:j+1]
        else:
            return self.long_str(i+1,j,s) if len(self.long_str(i+1,j,s)) >=len(self.long_str(i,j-1,s)) else self.long_str(i,j-1,s)

    def longestPalindrome(self, s):
        i =0
        j = len(s)-1
        return self.long_str(i,j,s)
str = "abcdasdfghjkldcba"
solution = Solution().longestPalindrome(str)
print("solution",solution)


