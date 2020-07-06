# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class TreeNode:
    def __init__(self,x):
        self.value = x
        self.left = None
        self.right = None

class Solution:
    def diameterOfBinaryTree(self, TreeNode):
        diameter = 0
        if TreeNode == None:
            return 0
        left_diameter = self.diameterOfBinaryTree(TreeNode.left)
        right_diameter = self.diameterOfBinaryTree(TreeNode.right)
        diameter = max(left_diameter + right_diameter,diameter)

    def path_depth(self,TreeNode):
        if TreeNode ==None :
            return 0
        l_path,l_depth = self.path_depth(TreeNode.left)
        r_path,r_depth = self.path_depth(TreeNode.right)
        max_path = max(l_path,r_path,1+l_depth+r_depth)


