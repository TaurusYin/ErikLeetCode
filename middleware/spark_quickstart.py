import os
import pyspark
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("test_SamShare").setMaster("local[4]")
sc = SparkContext(conf=conf)

# 使用 parallelize方法直接实例化一个RDD
rdd = sc.parallelize(range(1,11),4) # 这里的 4 指的是分区数量
rdd.take(100)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


"""
----------------------------------------------
                Transform算子解析
----------------------------------------------
"""
# 以下的操作由于是Transform操作，因为我们需要在最后加上一个collect算子用来触发计算。
# 1. map: 和python差不多，map转换就是对每一个元素进行一个映射
rdd = sc.parallelize(range(1, 11), 4)
rdd_map = rdd.map(lambda x: x*2)
print("原始数据：", rdd.collect())
print("扩大2倍：", rdd_map.collect())
# 原始数据： [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 扩大2倍： [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# 2. flatMap: 这个相比于map多一个flat（压平）操作，顾名思义就是要把高维的数组变成一维
rdd2 = sc.parallelize(["hello SamShare", "hello PySpark"])
print("原始数据：", rdd2.collect())
print("直接split之后的map结果：", rdd2.map(lambda x: x.split(" ")).collect())
print("直接split之后的flatMap结果：", rdd2.flatMap(lambda x: x.split(" ")).collect())
# 直接split之后的map结果： [['hello', 'SamShare'], ['hello', 'PySpark']]
# 直接split之后的flatMap结果： ['hello', 'SamShare', 'hello', 'PySpark']

# 3. filter: 过滤数据
rdd = sc.parallelize(range(1, 11), 4)
print("原始数据：", rdd.collect())
print("过滤奇数：", rdd.filter(lambda x: x % 2 == 0).collect())
# 原始数据： [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 过滤奇数： [2, 4, 6, 8, 10]

# 4. distinct: 去重元素
rdd = sc.parallelize([2, 2, 4, 8, 8, 8, 8, 16, 32, 32])
print("原始数据：", rdd.collect())
print("去重数据：", rdd.distinct().collect())
# 原始数据： [2, 2, 4, 8, 8, 8, 8, 16, 32, 32]
# 去重数据： [4, 8, 16, 32, 2]

# 5. reduceByKey: 根据key来映射数据
from operator import add
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
print("原始数据：", rdd.collect())
print("原始数据：", rdd.reduceByKey(add).collect())
# 原始数据： [('a', 1), ('b', 1), ('a', 1)]
# 原始数据： [('b', 1), ('a', 2)]

# 6. mapPartitions: 根据分区内的数据进行映射操作
rdd = sc.parallelize([1, 2, 3, 4], 2)
def f(iterator):
    yield sum(iterator)
print(rdd.collect())
print(rdd.mapPartitions(f).collect())
# [1, 2, 3, 4]
# [3, 7]

# 7. sortBy: 根据规则进行排序
tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
print(sc.parallelize(tmp).sortBy(lambda x: x[0]).collect())
print(sc.parallelize(tmp).sortBy(lambda x: x[1]).collect())
# [('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
# [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]

# 8. subtract: 数据集相减, Return each value in self that is not contained in other.
x = sc.parallelize([("a", 1), ("b", 4), ("b", 5), ("a", 3)])
y = sc.parallelize([("a", 3), ("c", None)])
print(sorted(x.subtract(y).collect()))
# [('a', 1), ('b', 4), ('b', 5)]

# 9. union: 合并两个RDD
rdd = sc.parallelize([1, 1, 2, 3])
print(rdd.union(rdd).collect())
# [1, 1, 2, 3, 1, 1, 2, 3]

# 10. intersection: 取两个RDD的交集，同时有去重的功效
rdd1 = sc.parallelize([1, 10, 2, 3, 4, 5, 2, 3])
rdd2 = sc.parallelize([1, 6, 2, 3, 7, 8])
print(rdd1.intersection(rdd2).collect())
# [1, 2, 3]

# 11. cartesian: 生成笛卡尔积
rdd = sc.parallelize([1, 2])
print(sorted(rdd.cartesian(rdd).collect()))
# [(1, 1), (1, 2), (2, 1), (2, 2)]

# 12. zip: 拉链合并，需要两个RDD具有相同的长度以及分区数量
x = sc.parallelize(range(0, 5))
y = sc.parallelize(range(1000, 1005))
print(x.collect())
print(y.collect())
print(x.zip(y).collect())
# [0, 1, 2, 3, 4]
# [1000, 1001, 1002, 1003, 1004]
# [(0, 1000), (1, 1001), (2, 1002), (3, 1003), (4, 1004)]

# 13. zipWithIndex: 将RDD和一个从0开始的递增序列按照拉链方式连接。
rdd_name = sc.parallelize(["LiLei", "Hanmeimei", "Lily", "Lucy", "Ann", "Dachui", "RuHua"])
rdd_index = rdd_name.zipWithIndex()
print(rdd_index.collect())
# [('LiLei', 0), ('Hanmeimei', 1), ('Lily', 2), ('Lucy', 3), ('Ann', 4), ('Dachui', 5), ('RuHua', 6)]

# 14. groupByKey: 按照key来聚合数据
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
print(rdd.collect())
print(sorted(rdd.groupByKey().mapValues(len).collect()))
print(sorted(rdd.groupByKey().mapValues(list).collect()))
# [('a', 1), ('b', 1), ('a', 1)]
# [('a', 2), ('b', 1)]
# [('a', [1, 1]), ('b', [1])]

# 15. sortByKey:
tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
print(sc.parallelize(tmp).sortByKey(True, 1).collect())
# [('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]

# 16. join:
x = sc.parallelize([("a", 1), ("b", 4)])
y = sc.parallelize([("a", 2), ("a", 3)])
print(sorted(x.join(y).collect()))
# [('a', (1, 2)), ('a', (1, 3))]

# 17. leftOuterJoin/rightOuterJoin
x = sc.parallelize([("a", 1), ("b", 4)])
y = sc.parallelize([("a", 2)])
print(sorted(x.leftOuterJoin(y).collect()))
# [('a', (1, 2)), ('b', (4, None))]

"""
----------------------------------------------
                Action算子解析
----------------------------------------------
"""
# 1. collect: 指的是把数据都汇集到driver端，便于后续的操作
rdd = sc.parallelize(range(0, 5))
rdd_collect = rdd.collect()
print(rdd_collect)
# [0, 1, 2, 3, 4]

# 2. first: 取第一个元素
sc.parallelize([2, 3, 4]).first()
# 2

# 3. collectAsMap: 转换为dict，使用这个要注意了，不要对大数据用，不然全部载入到driver端会爆内存
m = sc.parallelize([(1, 2), (3, 4)]).collectAsMap()
m
# {1: 2, 3: 4}

# 4. reduce: 逐步对两个元素进行操作,分成5组
rdd = sc.parallelize(range(10),5)
print(rdd.reduce(lambda x,y:x+y))
# 45

# 5. countByKey/countByValue:
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
print(sorted(rdd.countByKey().items()))
print(sorted(rdd.countByValue().items()))
# [('a', 2), ('b', 1)]
# [(('a', 1), 2), (('b', 1), 1)]

# 6. take: 相当于取几个数据到driver端
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
print(rdd.take(5))
# [('a', 1), ('b', 1), ('a', 1)]

# 7. saveAsTextFile: 保存rdd成text文件到本地
text_file = "./data/rdd.txt"
rdd = sc.parallelize(range(5))
rdd.saveAsTextFile(text_file)

# 8. takeSample: 随机取数
rdd = sc.textFile("./test/data/hello_samshare.txt", 4)  # 这里的 4 指的是分区数量
rdd_sample = rdd.takeSample(True, 2, 0)  # withReplacement 参数1：代表是否是有放回抽样
rdd_sample

# 9. foreach: 对每一个元素执行某种操作，不生成新的RDD
rdd = sc.parallelize(range(10), 5)
accum = sc.accumulator(0)
rdd.foreach(lambda x: accum.add(x))
print(accum.value)
# 45