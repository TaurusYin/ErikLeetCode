def merge_sort(nums, low, high):
    if low == high:
        return
    mid = (low + high) // 2
    merge_sort(nums, low, mid)
    merge_sort(nums, mid + 1, high)
    merge(nums, low, mid, high)
    return nums


def merge(nums, low, mid, high):
    i, j, tmp = low, mid + 1, []
    while i <= mid and j <= high:
        print('{},{}'.format(nums[i:mid + 1], nums[mid + 1:j + 1]))
        if nums[i] <= nums[j]:  # 左半区第一个剩余元素更小
            tmp.append(nums[i])
            i += 1
        else:
            tmp.append(nums[j])
            j += 1
        print('tmp:{}'.format(tmp))
    if i <= mid: # 右半区用完了，左半区直接搬过去
        tmp.extend(nums[i:mid + 1])
    if j <= high: # 左半区用完了，右半区直接搬过去
        tmp.extend(nums[j:high + 1])
    nums[low:high + 1] = tmp # 把合并后的数组拷回原来的数组

# https://leetcode.cn/problems/advantage-shuffle/
def advantageCount(A, B):
        sortedA = sorted(A)
        sortedB = sorted(B)

        # assigned[b] = list of a that are assigned to beat b
        # remaining = list of a that are not assigned to any b
        assigned = {b: [] for b in B}
        remaining = []

        # populate (assigned, remaining) appropriately
        # sortedB[j] is always the smallest unassigned element in B
        j = 0
        for a in sortedA:
            if a > sortedB[j]:
                assigned[sortedB[j]].append(a)
                j += 1
            else:
                remaining.append(a)

        # Reconstruct the answer from annotations (assigned, remaining)
        return [assigned[b].pop() if assigned[b] else remaining.pop()
                for b in B]
nums1 = [2,7,11,15] ; nums2 = [1,10,4,11]
advantageCount(nums1,nums2)


if __name__ == '__main__':
    nums = [100, 2, 3, 4, 10, 40]
    nums = [1, 3, 2, 3, 1]
    print(merge_sort(nums, 0, len(nums) - 1))
