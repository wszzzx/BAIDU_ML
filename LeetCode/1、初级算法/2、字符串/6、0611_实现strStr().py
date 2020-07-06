class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle == "":
            return 0
        h_len = len(haystack)
        n_len = len(needle)
        if(h_len < n_len):
            return False
        for i in range(h_len - n_len+1):
            temp = haystack[i:i+n_len]
            if temp == needle:
                return i
        return -1
str1 = "hello"
str2 = "ll"

ans = Solution().strStr(str1,str2)
print("ans",ans)




