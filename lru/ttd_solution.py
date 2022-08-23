from pprint import pprint

from lru.lru_cache import LRUCache
import threading


class NWayCache:
    def __init__(self, cache_size, n_items, algorithm='LRU'):
        self.cache_size = cache_size  # size of cache 20
        self.n_items = n_items  # number of cache items(n-sized sets, n-way)
        self.n_sets = int(self.cache_size / self.n_items)
        self.algorithm = algorithm
        self.cache = {}
        self.lock = threading.RLock()
        for i in range(0, self.n_sets):
            if algorithm == 'LRU':
                cache_method = LRUCache(capacity=self.n_items, n_sets=i)
            else:
                cache_method = LRUCache(capacity=self.n_items, n_sets=i)
            self.cache[i] = cache_method

    def get_set_index(self, key):
        index = key % self.n_sets  # hashcode
        return index

    def get(self, key):
        # 加锁
        self.lock.acquire()
        try:
            return self.cache[self.get_set_index(key)].get(key)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
        finally:
            self.lock.release()


    def put(self, key, val):
        self.lock.acquire()
        try:
            self.cache[self.get_set_index(key)].put(key, val)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
        finally:
            self.lock.release()
        return




c = NWayCache(cache_size=16, n_items=2, algorithm='LRU')
c.put(1, 1)
c.put(2, 2)
x = c.get(1)
c.put(3, 3)
x = c.get(2)

print()
