# 1. 尽可能复用同一个RDD，避免重复创建，并且适当持久化数据
# 最低级写法，相同数据集重复创建。
rdd1 = sc.textFile("./test/data/hello_samshare.txt", 4) # 这里的 4 指的是分区数量
rdd2 = sc.textFile("./test/data/hello_samshare.txt", 4) # 这里的 4 指的是分区数量
print(rdd1.take(10))
print(rdd2.map(lambda x:x[0:1]).take(10))

# 稍微进阶一些，复用相同数据集，但因中间结果没有缓存，数据会重复计算
rdd1 = sc.textFile("./test/data/hello_samshare.txt", 4) # 这里的 4 指的是分区数量
print(rdd1.take(10))
print(rdd1.map(lambda x:x[0:1]).take(10))

# 相对比较高效，使用缓存来持久化数据
rdd = sc.parallelize(range(1, 11), 4).cache()  # 或者persist()
rdd_map = rdd.map(lambda x: x*2)
rdd_reduce = rdd.reduce(lambda x, y: x+y)
print(rdd_map.take(10))
print(rdd_reduce)




# 原则2：尽量避免使用低性能算子
rdd1 = sc.parallelize([('A1', 211), ('A1', 212), ('A2', 22), ('A4', 24), ('A5', 25)])
rdd2 = sc.parallelize([('A1', 11), ('A2', 12), ('A3', 13), ('A4', 14)])
# 低效的写法，也是传统的写法，直接join
rdd_join = rdd1.join(rdd2)
print(rdd_join.collect())
# [('A4', (24, 14)), ('A2', (22, 12)), ('A1', (211, 11)), ('A1', (212, 11))]
rdd_left_join = rdd1.leftOuterJoin(rdd2)
print(rdd_left_join.collect())
# [('A4', (24, 14)), ('A2', (22, 12)), ('A5', (25, None)), ('A1', (211, 11)), ('A1', (212, 11))]
rdd_full_join = rdd1.fullOuterJoin(rdd2)
print(rdd_full_join.collect())
# [('A4', (24, 14)), ('A3', (None, 13)), ('A2', (22, 12)), ('A5', (25, None)), ('A1', (211, 11)), ('A1', (212, 11))]

# 高效的写法，使用广播+map来实现相同效果
# tips1: 这里需要注意的是，用来broadcast的RDD不可以太大，最好不要超过1G
# tips2: 这里需要注意的是，用来broadcast的RDD不可以有重复的key的
rdd1 = sc.parallelize([('A1', 11), ('A2', 12), ('A3', 13), ('A4', 14)])
rdd2 = sc.parallelize([('A1', 211), ('A1', 212), ('A2', 22), ('A4', 24), ('A5', 25)])

# step1： 先将小表进行广播，也就是collect到driver端，然后广播到每个Executor中去。
rdd_small_bc = sc.broadcast(rdd1.collect())

# step2：从Executor中获取存入字典便于后续map操作
rdd_small_dict = dict(rdd_small_bc.value)

# step3：定义join方法
def broadcast_join(line, rdd_small_dict, join_type):
    k = line[0]
    v = line[1]
    small_table_v = rdd_small_dict[k] if k in rdd_small_dict else None
    if join_type == 'join':
        return (k, (v, small_table_v)) if k in rdd_small_dict else None
    elif join_type == 'left_join':
        return (k, (v, small_table_v if small_table_v is not None else None))
    else:
        print("not support join type!")

# step4：使用 map 实现 两个表join的功能
rdd_join = rdd2.map(lambda line: broadcast_join(line, rdd_small_dict, "join")).filter(lambda line: line is not None)
rdd_left_join = rdd2.map(lambda line: broadcast_join(line, rdd_small_dict, "left_join")).filter(lambda line: line is not None)
print(rdd_join.collect())
print(rdd_left_join.collect())
# [('A1', (211, 11)), ('A1', (212, 11)), ('A2', (22, 12)), ('A4', (24, 14))]
# [('A1', (211, 11)), ('A1', (212, 11)), ('A2', (22, 12)), ('A4', (24, 14)), ('A5', (25, None))]

