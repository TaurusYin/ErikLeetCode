
arr = [[0] * 6 for _ in range(10)]
# a = [dog for dog in range(100)]
a = list(range(100))

# 如果列表是一个等差数列，直接用range更快了

b = [i + 1 for i in range(100)]
b_ = list(range(1, 101))

c = [i for i in range(100) if i % 2 == 0]
c_ = list(range(0, 100, 2))


import abc
class Person(metaclass=abc.ABCMeta):
    def __init__(self, name, gender, age, birthday):
        self.name: str = name
        self.gender: str = gender
        self.age: int = age
        self.birthday = birthday

    def test(self):
        # if date == self.birthday:
        #     self.age += 1
        ...

    @abc.abstractmethod
    def say(self):
        ...


class A:
    a = 5
    _a = 5
    __a = 5
    def __init__(self):
        self.b = 5
        self._b = 5
        self.__b = 5

    def test(self):
        print(f"{self.b},{self._b},{self.__b}")
        print(f"{self.f()},{self._f()},{self.__f()}")

    @classmethod
    def test2(cls):
        print(f"{cls.a},{cls._a},{cls.__a}")
        print(f"{cls.g()},{cls._g()},{cls.__g()}")

    def f(self):
        ...

    def _f(self):
        ...

    def __f(self):
        ...

    @classmethod
    def g(cls):
        ...

    @classmethod
    def _g(cls):
        ...

    @classmethod
    def __g(cls):
        ...

if __name__ == '__main__':
    a = A()
    a.test()
    a.test2()
