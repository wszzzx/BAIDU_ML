# 字符串压缩。利用字符重复出现的次数，编写一种方法，实现基本的字符串压缩功能。比如，字符串aabcccccaaa会变为a2b1c5a3。若“压缩”后的字符串没有变短，则返回原先的字符串。你可以假设字符串中只包含大小写英文字母（a至z）。
#
# 示例1:
#  输入："aabcccccaaa"
#  输出："a2b1c5a3"

# 示例2:
#  输入："abbccd"
#  输出："abbccd"
#  解释："abbccd"压缩后为"a1b2c2d1"，比原字符串长度更长。

# a
class Solution:
    # def compressString(self, s: str) -> str:
    #     s_len = len(s)
    #     compress_s = ""
    #     countnum = 0
    #     for i in range(s_len):
    #         if countnum == 0:
    #             compress_s += s[i]
    #         countnum += 1
    #         if i < s_len -1 :
    #             if s[i] != s[i+1]:
    #                 compress_s +=str(countnum)
    #                 countnum = 0
    #         else:
    #             compress_s += str(countnum)
    #     return compress_s if len(compress_s) < len(s) else s
    def compressString(self, s: str) -> str:
        if not s:
            return ""
        ch = s[0]
        ans = ""
        chc = 0
        for c in s:
            if c != ch:
                ans += ch + str(chc)
                ch = c
                chc = 1
            else:
                chc += 1
        ans += ch + str(chc)
        return ans if len(ans) < len(s) else s




solution = Solution()
s = "aaabbcc"
result = solution.compressString(s)
print("result",result)