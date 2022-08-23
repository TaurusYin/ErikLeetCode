from functools import lru_cache
import datetime

def mannual_cache(func):
    data = {}
    def wrapper(n):
        if n in data:
            return data[n]
        else:
            res = func(n)
            data[n] = res
            return res
    return wrapper

@lru_cache
def test(a, b):
    print('开始计算a+b的值...')
    return a + b

@mannual_cache
def fibonacci(num):
	# 不使用缓存时，会重复执行函数
    return num if num < 2 else fibonacci(num - 1) + fibonacci(num - 2)


print(test(1, 2))
print(test(1, 2))

start = datetime.datetime.now()
print(fibonacci(40))
end = datetime.datetime.now()
print('执行时间', end - start)