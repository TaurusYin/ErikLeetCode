import pyspark
# DataFrame.toPandas
# 把SparkDataFrame转为 Pandas的DataFrame
df_pandas = df.toPandas()

# DataFrame.rdd
# 把SparkDataFrame转为rdd，这样子可以用rdd的语法来操作数据
# df.rdd
import pandas as pd

def _map_to_pandas(rdds):
    return [pd.DataFrame(list(rdds))]


def toPandas(df, n_partitions=None):
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand


df_new_pandas = toPandas(df, 2)

for idx, row in df_new_pandas.iterrows():
    print(row)

df_spark = spark.createDataFrame(df_new_pandas)
df_spark.show()






# spark_df -> RDD
df = df.select(["user", "item", "score"])
rdd = df.map(tuple)

df = df.select(["user", "item", "score"])
rdd = df.rdd
rdd1 = rdd.map(lambda p:(p["user"], p["item"], p["score"]))


# RDD -> spark_df
rdd = sc.parallelize([('aliance',5),('horde', 3)])
df = rdd.toDF()
columns = ['name', 'score']
df = rdd.toDF(columns)

# RDD -> spark_df
data = [('Alex','male',3),('Nancy','female',6),['Jack','male',9]] # mixed
rdd_ = spark.sparkContext.parallelize(data)

# schema
schema = StructType([
        # true代表不为空
        StructField("name", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("num", StringType(), True)
    ])
df = spark.createDataFrame(rdd_, schema=schema)  # working when the struct of data is same.
print(df.show())
"""
对一个已有的DataFrame，以下方法可以查看当前DataFrame的schema信息

df.printSchema() # 打印schema信息
# root
#  |-- age: integer (nullable = true)
#  |-- name: string (nullable = true)
# <BLANKLINE>
schema = df.schema # 直接得到当前df的schema，是StructType类型。
# StructType(List(StructField(age,IntegerType,true),
#                 StructField(name,StringType,true)))
  可以发现StructField对象相当于一个字段，它有这个字段的name,dataType和nullable。
case class StructType(fields: Array[StructField]) extends DataType with Seq[StructField] {}
  而从StructType的定义来看，StructType相当于一个集合，里面的元素都是StructField类型。而所谓的schema信息就是一个StructField的集合StructType。

"""
# schema helper
from pyspark.sql.types import *
from helper import StructCollect
cols = ['user', 'scores']
df = sc.parallelize([['123', [5, 4]], ['fs', []], ['fsd', [2, 3, 4]]]).toDF(cols)
schema = StructCollect(df)
print(schema.names)
# get:get a field
print(schema.get("user"))
# merge:合并两个dataframe的schema，得到新的schema
new_df = sc.parallelize([[123]]).toDF(["height"])
schema = schema.merge(new_df)
print(schema)
# append:添加新的fields，得到新的schema
schema = schema.append("items:array<int>, education:string")
print(schema)