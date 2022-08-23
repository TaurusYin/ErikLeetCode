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




if __name__ == '__main__':
    nums = [100, 2, 3, 4, 10, 40]
    nums = [1, 3, 2, 3, 1]
    print(merge_sort(nums, 0, len(nums) - 1))
