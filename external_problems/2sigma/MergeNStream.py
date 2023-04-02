from sortedcontainers import SortedList
import time

def merge_streams(streams):
    # 创建一个空的sortedlist，并按照时间戳排序
    merged_list = SortedList(key=lambda x: x[1])

    # 将所有流中的元素添加到sortedlist中
    for stream in streams:
        for element in stream:
            merged_list.add(element)

    # 按照时间戳合并元素
    result = []
    prev_timestamp = None
    for element in merged_list:
        delta, timestamp = element
        if timestamp != prev_timestamp:
            result.append((delta, timestamp))
            prev_timestamp = timestamp
        else:
            result[-1] = (result[-1][0] + delta, timestamp)

    return result

# 创建两个流，每个流包含三个元素
stream1 = [(1, time.time()), (2, time.time()+1), (3, time.time()+2)]
stream2 = [(4, time.time()), (5, time.time()+1), (6, time.time()+2)]

# 将两个流传递给merge_streams函数，并打印结果
result = merge_streams([stream1, stream2])
print(result)
