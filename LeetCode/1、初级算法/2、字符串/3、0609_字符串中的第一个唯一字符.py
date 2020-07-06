class Solution:
    def firstUniqChar(self, s: str) -> int:
        s_length = len(s)
        ans_list = []
        ans_dict = {}
        for i in range(s_length):
            s_ch = s[i]
            if s_ch not in ans_dict.keys():
                ans_dict[s_ch] = i
                ans_list.append(i)
            else:
                index = ans_dict[s_ch]
                if index in ans_list:
                    ans_list.remove(index)
        return ans_list[0] if len(ans_list) >0 else -1

s = "abcabc"
ans = Solution().firstUniqChar(s)
print("ans",ans)



