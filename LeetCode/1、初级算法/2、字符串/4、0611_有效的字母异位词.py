class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        s_dict = {}
        t_dict = {}
        for ch in s:
            if ch not in s_dict.keys():
                s_dict[ch] =1
            else:
                s_dict[ch] +=1
        for ch in t:
            if ch not in t_dict.keys():
                t_dict[ch] =1
            else:
                t_dict[ch] +=1
        if len(s_dict) !=len(t_dict):
            return False
        else:
            for key in s_dict.keys():
                if key not in t_dict.keys():
                    return False
                elif s_dict[key] != t_dict[key]:
                    return False
        return True

s = "abc"
t = "cba"
flag = Solution().isAnagram(s,t)
print("flag",flag)