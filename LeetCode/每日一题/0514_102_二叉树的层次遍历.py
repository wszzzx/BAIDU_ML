# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def levelOrder(self, root) :
        return_lists = []
        return_list = []
        if root == None:
            return return_lists
        else:
            return_list.append(root)
            while(len(return_list)>0):
                values = []
                root_list = []
                for root in return_list:
                    if root!=None:
                        values.append(root.val)
                        if root.left !=None:
                            root_list.append(root.left)
                        if root.right != None:
                            root_list.append(root.right)
                return_list = root_list
                return_lists.append(values)
        return return_lists

treenode = TreeNode(3)
treenode.left= TreeNode(9)
treenode.right = TreeNode(8)
treenode.right.left = TreeNode(7)
treenode.right.right = TreeNode(6)
treenode.right.right.right = TreeNode(6)


solution = Solution()
return_lists = solution.levelOrder(None)
print("return_lists",return_lists)








