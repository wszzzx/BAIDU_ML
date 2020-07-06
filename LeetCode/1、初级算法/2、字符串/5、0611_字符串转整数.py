class Solution:
    def myAtoi(self, str: str) -> int:
        first_flag = True
        valid =1
        ans =""

        for ch in str:
            if first_flag ==True:
                if ch !=' ':
                    first_flag = False
                    if ch =="+":
                        valid =1
                    elif ch =="-":
                        valid = -1
                    elif ch.isdigit():
                        ans +=ch
                    else:
                        return 0
            else:
                if ch.isdigit():
                    ans +=ch
                else:
                    break
        if ans =="":
            return 0
        ans_num = int(ans)*valid
        if ans_num < -2**31:
            return -2**31
        elif ans_num >2**31 -1:
            return  2**31-1
        else:
            return ans_num
str ="+"
ans = Solution().myAtoi(str)
print("ans",ans)







