from arrangement import number_problem
from functools import cache

blocks = [
    {'block_id': 0, 'price': 180, 'guests': 1},
    {'block_id': 1, 'price': 280, 'guests': 2},
    {'block_id': 2, 'price': 310, 'guests': 2},
    {'block_id': 3, 'price': 450, 'guests': 3},

]

number_of_guests = 7


def search(blocks, number_of_guests):
    print(blocks)
    print(number_of_guests)
    combinations, path, cur_guests, min_price = [], [], 0, number_problem.inf
    # @cache
    def backtrack(blocks, start_index, cur_guests, cur_price):
        nonlocal min_price
        combinations.append(path[:])  # 收集子集，要放在终止添加的上面，否则会漏掉自己
        print('combinations: {}, path:{}'.format(combinations, path[:]))
        for i in range(start_index, len(blocks)):  # 当startIndex已经大于数组的长度了，就终止了，for循环本来也结束了，所以不需要终止条件
            print('local path:{}'.format(path))
            if cur_guests < number_of_guests:
                cur_guests += blocks[i]['guests']
                cur_price += blocks[i]['price']
                path.append((blocks[i], cur_price))
                if cur_price < min_price and cur_guests >= number_of_guests:
                    min_price = cur_price
                    print('min_price: {}'.format(min_price))
                backtrack(blocks, i + 1, cur_guests, cur_price)  # 递归
                path.pop() # 回溯
                cur_guests -= blocks[i]['guests']
                cur_price -= blocks[i]['price']
        return min_price

    min_price = backtrack(blocks, 0, cur_guests, 0)
    [print('combinations: {}'.format(elem)) for elem in combinations]
    return min_price


search(blocks, number_of_guests)
