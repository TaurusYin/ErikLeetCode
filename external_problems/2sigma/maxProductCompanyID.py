"""
题目要求：

给定一组三元组列表，每个三元组包含两个朋友ID（friendId）和一个公司ID（companyId）。要求找出在同一家公司中具有最长朋友关系链的两个朋友ID，使得这两个朋友ID的乘积最大。朋友关系链具有传递性，即如果朋友1和朋友2是朋友，朋友2和朋友3也是朋友，那么朋友1和朋友3也被认为是朋友。

示例：

输入：
friendsInACompany = [
[1, 2, 1],
[2, 7, 1],
[4, 1, 1],
[1, 4, 2],
[3, 4, 2]
]

输出：
14

解释：
在公司1中，朋友关系链为 1-2-7-4，朋友ID的乘积最大为 74 = 28。
在公司2中，朋友关系链为 1-4-3，朋友ID的乘积最大为 43 = 12。

所以返回最大乘积 28。
"""

from collections import defaultdict


class UnionFind:
    def __init__(self, friends_ids):
        # 根据传入的 friends_ids 可迭代集合，创建一个字典。初始时，每个朋友 ID 作为它自己的根节点，表示它们各自独立，尚未与其他朋友 ID 建立连接。
        self.root = {friend_id: friend_id for friend_id in friends_ids}
        # 它根据传入的 friends_ids 可迭代集合，创建一个字典，将每个朋友 ID 的分组大小初始化为 1。这是因为在初始时，每个朋友 ID 都是独立的，分组只包含它们自己，所以分组大小为 1。
        self.size = {friend_id: 1 for friend_id in friends_ids}
        self.part = len(friends_ids)  # 联通分量个数

    def find(self, x):
        if x != self.root[x]:
            origin = self.root[x]
            self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:  # 找到环
            return True
        # parent_id 小的为父节点， 默认 y是x的父节点
        if root_x <= root_y:  # 没找到环，更新两个节点的parent为其中最小值
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # 将非根节点的秩赋0
        self.size[root_x] = 0
        self.part -= 1
        return False

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        # 获取每个根节点对应的组
        part = defaultdict(list)
        for friend_id in self.root:
            part[self.find(friend_id)].append(friend_id)
        return part

    def get_root_size(self):
        # 获取每个根节点对应的组大小
        size = defaultdict(int)
        for friend_id in self.root:
            size[self.find(friend_id)] = self.size[self.find(friend_id)]
        return size


from collections import defaultdict


class Solution:
    def __init__(self):
        self.max_size = 0
        self.max_groups = []

    def max_product_friends_group(self, friends_in_a_company):
        friend_in_same_cpy = defaultdict(list)
        max_product = 0

        # 将朋友关系按公司分组
        for frds_in_a_cpy in friends_in_a_company:
            company_id = frds_in_a_cpy[2]
            friend_in_same_cpy[company_id].append(frds_in_a_cpy[:2])

        # 对于每个公司，计算最大朋友关系链
        for frds_in_same_company in friend_in_same_cpy.values():
            friend_ids = set()
            for pair in frds_in_same_company:
                friend_ids |= set(pair)
            uf = UnionFind(friend_ids)
            for x, y in frds_in_same_company:
                uf.union(x, y)

            groups = uf.get_root_part()
            max_group = max(groups.values(), key=len)
            if len(max_group) == self.max_size:
                self.max_groups.append(max_group)
            elif len(max_group) > self.max_size:
                self.max_groups = [max_group]
                self.max_size = len(max_group)

        # 计算最大乘积
        for max_group in self.max_groups:
            max_group.sort(reverse=True)
            max_product = max(max_product, max_group[0] * max_group[1])

        return max_product


if __name__ == "__main__":
    friends = [
        [1, 2, 1],
        [2, 7, 1],
        [4, 1, 1],
        [1, 4, 2],
        [3, 4, 2]
    ]
    solution = Solution()
    print(solution.max_product_friends_group(friends))
