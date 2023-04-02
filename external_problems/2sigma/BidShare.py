import functools
def getUnallottedUsers(bids, totalShares):
    # 定义比较函数
    def bidComparator(a, b):
        one = a[2]
        two = b[2]
        if one == two:
            tone = a[3]
            ttwo = b[3]
            return tone - ttwo
        return two - one

    # 对竞标进行排序
    bids.sort(key=functools.cmp_to_key(bidComparator))
    id_l = 0
    users_set = set()
    while id_l < len(bids) and totalShares > 0:
        cur_price = bids[id_l][2]
        id_r = id_l
        while id_r < len(bids):
            tmp_price = bids[id_r][2]
            if tmp_price != cur_price:
                break
            id_r += 1

        change = True
        while change:
            change = False
            for i in range(id_l, id_r):
                if totalShares <= 0:
                    break
                cur_num = bids[i][1]
                if cur_num > 0:
                    totalShares -= 1
                    bids[i][1] = cur_num - 1
                    users_set.add(bids[i][0])
                    change = True

        id_l = id_r

    # 获取未分配的用户ID列表
    ans = []
    for i in range(len(bids)):
        uid = bids[i][0]
        if uid not in users_set:
            ans.append(uid)

    return ans



bids = [[1, 5, 10, 3], [2, 7, 12, 2], [3, 7, 10, 1], [4, 8, 9, 2]]
totalShares = 20
print(getUnallottedUsers(bids, totalShares))  # [1, 4]
