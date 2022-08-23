class Solution:
    def common_binary_search(self, nums, target: int):
        # [4, 5, 10, 10, 10, 13, 17, 21] target:10
        left, right = -1, len(nums)
        while (left + 1) != right:
            mid = (left + right) >> 1
            # if nums[mid] isBlue()
            if nums[mid] <= target:  # search last 10
                left = mid
            else:
                right = mid
        # return left or right
        return left

    '''
    该系列模板与标准模板有以下不同：

初始要判断空数组。
循环条件是：left < right，没有 = 号。所以退出循环后，left = right。只是我习惯使用 right。
mid 的取值以及区间的选取不同：某个区间可能包含 mid 的值。
往往排除某个区间比较容易思考，所以通过不断排除区间的方法得到最终值。剩余条件直接让 left 或 right 等于 mid 即可。
若是结果隐含了第一、首次、左边、最小等条件，则 mid 取下整，即 mid = left + right >> 1。
考虑收缩左边界，使用 left = mid + 1。
若是结果隐含了最后、右边、最大等条件，则 mid 取上整，即 mid = left + right + 1 >> 1。
考虑收缩右边界，使用 right = mid - 1。
循环结束后，还要进行条件的判断。

作者：繁星满天
链接：https://leetcode.cn/circle/article/bNaUjl/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
    '''

    # 查找可以通过访问无重复数组中的单个索引来确定的元素或条件
    def binary_target_search(self, nums, target: int):
        if not nums:
            return -1
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            print('left = {}, right = {}, mid = {}'.format(left, right, mid))
            if target == nums[mid]:
                return mid
            elif target > nums[mid]:
                left = mid + 1
            elif target < nums[mid]:
                right = mid - 1
        return -1

    def first_search(self, nums, target):
        if not nums:
            return -1
        left, right = 0, len(nums) - 1
        while left < right: # <= -> <
            mid = left + right >> 1
            print('left = {}, right = {}, mid = {}'.format(left, right, mid))
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid # right = mid - 1 -> right = mid
        if nums[right] == target:
            return right
        return -1




    def searchInsert(self, nums: list, target: int) -> int:
        # 返回大于等于 target 的索引，有可能是最后一个
        size = len(nums)
        # 特判
        if size == 0:
            return 0
        left = 0
        # 如果 target 比 nums里所有的数都大，则最后一个数的索引 + 1 就是候选值，因此，右边界应该是数组的长度
        right = size
        # 二分的逻辑一定要写对，否则会出现死循环或者数组下标越界
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                assert nums[mid] >= target
                # [1,5,7] 2
                right = mid
            # 调试语句
            print('left = {}, right = {}, mid = {}'.format(left, right, mid))
        return left


if __name__ == '__main__':
    nums = [4, 5, 10, 10, 10, 13, 17, 21]
    x = Solution().common_binary_search(nums=nums, target=10)
    x = Solution().binary_target_search(nums=nums, target=10)
    x = Solution().first_search(nums=nums, target=10)

    print(x)
