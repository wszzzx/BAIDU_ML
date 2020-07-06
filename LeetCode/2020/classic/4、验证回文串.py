class Solution:
    def isPalindrome(self, s):
        s = s.lower().strip()
        s_len = len(s)
        print(s)
        i = 0
        j = s_len -1
        while(i<j):
            while(i<j):
                start_char = s[i]
                if start_char.isdigit() or start_char.isalpha():
                    break
                else:
                    i+=1

            while(i<j):
                end_char = s[j]
                if end_char.isdigit() or end_char.isalpha():
                    break
                else:
                    j-=1
            if i<j and start_char != end_char:
                return  False
            else:
                i+=1
                j-=1
        return True


s = " apG0i4maAs::sA0m4i0Gp0"
solution = Solution()
flag = solution.isPalindrome(s)
print(flag)




