class TreeNode():
    def __init__(self,x):
        self.value = x
        self.left = None
        self.right = None
class Solution:
    def levelOrder(self, root):
        ans_list = []
        TreeNode_list = []
        if root == None:
            return ans_list
        else:
            TreeNode_list.append(root)
        while(len(TreeNode_list)>0):
            temp_ans = []
            temp_list = []
            # print("TreeNode_list",TreeNode_list)
            for treenode in TreeNode_list:
                # print("Node",Node,type(Node))
                if treenode != None:
                    if treenode.left != None:
                        temp_list.append(treenode.left)
                    if treenode.right != None:
                        temp_list.append(treenode.right)
                    temp_ans.append(treenode.value)
            ans_list.append(temp_ans)
            TreeNode_list = temp_list
        return ans_list
solution = Solution()
treenode = TreeNode(3)
treenode.right = TreeNode(20)
treenode.left = TreeNode(9)
treenode.right.left = TreeNode(15)
treenode.right.right = TreeNode(7)
print("treenode",treenode.left)
list = solution.levelOrder(treenode)
print("list",list)





