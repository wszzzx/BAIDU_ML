class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs)==0:
            return ""
        shortest_str = ""
        short =len(strs[0])
        for str in strs:
            if str =="":
                return ""





