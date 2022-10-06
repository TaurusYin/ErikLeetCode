# https://xie.infoq.cn/article/e7182d18df48bc26eeb30b207
# /bin/python
# -*- coding: utf-8 -*-
import bisect
import hashlib

from numpy.compat import long


def get_hash(raw_str):
    """将字符串映射到2^32的数字中"""
    md5_str = hashlib.md5(raw_str).hexdigest()
    return long(md5_str, 16)


class CacheNode(object):
    """缓存节点类，负责记录缓存服务器信息，以及发送缓存服务器方法"""

    def __init__(self, ip):
        self.ip = ip

    def send(self, request):
        """发送到对应的cache
        Args:
            request：需要转发的request信息
        """
        # 假装有内容的亚子
        pass


class HashSeverMgr(object):
    """server管理类，给定ip，返回对应的server"""

    def __init__(self):
        self.cache_list = []
        self.cache_node = dict()
        self.virtual_num = 10

    def add_server(self, node):
        """添加缓存节点
        Args:
            node: 缓存节点类CacheNode，记录缓存server的香港信息。
        """
        node_hash = get_hash(node.ip)
        bisect.insort(self.cache_list, node_hash)
        self.cache_node[node_hash] = node

    def del_server(self, node):
        """删除缓存节点"""
        node_hash = get_hash(node.ip)
        self.cache_list.remove(node_hash)
        del self.cache_node[node_hash]

    """
    一致性 hash 解决了增加/下线 缓存服务器，可能引起的缓存击穿等问题。但是当缓存服务器数量较少时，出现数据倾斜等问题。
为了解决这个问题，产生了基于虚拟节点的一致性 hash 算法。即，一个缓存服务器映射到 hash 环上的多个地址。从而，减少数据倾斜的情况。
    """
    def _add_server(self, node):
        """添加缓存节点
        Args:
            node: 缓存节点类CacheNode，记录缓存server的香港信息。
        """
        for index in range(0, self.virtual_num):
            node_hash = get_hash("%s_%s" % (node.ip, index))
            bisect.insort(self.cache_list, node_hash)
            self.cache_node[node_hash] = node

    def _del_server(self, node):
        """删除缓存节点"""
        for index in range(0, self.virtual_num):
            node_hash = get_hash("%s_%s" % (node.ip, index))
            self.cache_list.remove(node_hash)
            del self.cache_node[node_hash]

    def get_server(self, source_key):
        """获取目标缓存节点，确定待转发的目标缓存server"""
        key_hash = get_hash(source_key)
        index = bisect.bisect_left(self.cache_list, key_hash)
        index = index % len(self.cache_list)  # 若比最大的node hash还大，分发给第一个node
        return self.cache_node[self.cache_list[index]]


if __name__ == "__main__":
    cache_ips = [
        "1.2.3.4",
        "2.3.4.5",
        "3.4.5.6"
    ]
    source_key = "1234567890"
    hash_mgr = HashSeverMgr()
    for ip in cache_ips:
        cache_node_temp = CacheNode(ip)
        hash_mgr.add_server(cache_node_temp)

    sended_node = hash_mgr.get_server(source_key)
    print(sended_node.ip)
