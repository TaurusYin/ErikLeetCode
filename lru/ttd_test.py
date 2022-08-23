import unittest
from lru.ttd_solution import NWayCache

from collections import OrderedDict
x = OrderedDict()


class SolutionTest(unittest.TestCase):
    def test_case(self):
        c = NWayCache(cache_size=2, n_items=2, algorithm='LRU')
        c.put(1, 1)
        c.put(2, 2)
        x = c.get(1)
        c.put(3, 3)
        x = c.get(2)
        print('res:'.format(x))
        self.assertEqual(x, -1)

class Dict(dict):

    def __init__(self, **kw):
        super().__init__(**kw)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Dict' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value


class TestDict(unittest.TestCase):

    def test_init(self):
        d = Dict(a=1, b='test')
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 'test')
        self.assertTrue(isinstance(d, dict))

    def test_key(self):
        d = Dict()
        d['key'] = 'value'
        self.assertEqual(d.key, 'value')

    def test_attr(self):
        d = Dict()
        d.key = 'value'
        self.assertTrue('key' in d)
        self.assertEqual(d['key'], 'value')

    def test_keyerror(self):
        d = Dict()
        with self.assertRaises(KeyError):
            value = d['empty']

    def test_attrerror(self):
        d = Dict()
        with self.assertRaises(AttributeError):
            value = d.empty
