# !/usr/bin/env python3
""" Several users reading a calender, but only a few users updating it """
"""
（1） 互斥条件：一个资源每次只能被一个进程使用。 
（2） 请求与保持条件：一个进程因请求资源而阻塞时，对已获得的资源保持不放。 
（3） 不剥夺条件:进程已获得的资源，在末使用完之前，不能强行剥夺。 
（4） 循环等待条件:若干进程之间形成一种头尾相接的循环等待资源关系。

不可重入锁：Synchronized +wait + notifyAll  ：不可重入锁：只要有资源拿到锁的时候，其他进程就会等待。
可重入锁 Reentrant Lock ： ReentrantLock await unLock : 可重入锁，线程判断是否是本线程上锁，如果是同一线程则 锁 count++。
"""
import threading
import time

class Account:
    # 定义构造器
    def __init__(self, account_no, balance):
        # 封装账户编号、账户余额的两个成员变量
        self.account_no = account_no
        self._balance = balance
        self.lock = threading.RLock()

    # 因为账户余额不允许随便修改，所以只为self._balance提供getter方法
    def getBalance(self):
        return self._balance
    # 提供一个线程安全的draw()方法来完成取钱操作
    def draw(self, draw_amount):
        # 加锁
        self.lock.acquire()
        try:
            # 账户余额大于取钱数目
            if self._balance >= draw_amount:
                # 吐出钞票
                print(threading.current_thread().name\
                    + "取钱成功！吐出钞票:" + str(draw_amount))
                time.sleep(0.001)
                # 修改余额
                self._balance -= draw_amount
                print("\t余额为: " + str(self._balance))
            else:
                print(threading.current_thread().name\
                    + "取钱失败！余额不足！")
        finally:
            # 修改完成，释放锁
            self.lock.release()




import threading
from readerwriterlock import rwlock

WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
today = 0
marker = rwlock.RWLockFair()


def calendar_reader(id_number):
    global today
    read_marker = marker.gen_rlock()
    name = 'Reader-' + str(id_number)
    while today < len(WEEKDAYS) - 1:
        read_marker.acquire()
        print(name, 'sees that today is', WEEKDAYS[today], '-read count:', read_marker.c_rw_lock.v_read_count)
        read_marker.release()


def calender_writer(id_number):
    global today
    write_marker = marker.gen_wlock()
    name = 'Write-' + str(id_number)
    while today < len(WEEKDAYS) - 1:
        write_marker.acquire()
        today = (today + 1) % 7
        print(name, 'updated date to ', WEEKDAYS[today])
        write_marker.release()


class Tasks:
    def __init__(self, func):
        self.func = func

    def _unwrap_multi_process(self, arg, **kwarg):
        return self.func(arg, **kwarg)

    def multiprocess(self, object_list, num_process=1):
        """
        耗cpu的操作，用多进程编程。
        对于io操作来说， 使用多线程编程。
        """
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=num_process)
        results = pool.map(self._unwrap_multi_process, object_list)
        results = list(zip(object_list, results))
        return results

    def __multiprocess__(self, object_list, num_process=1):
        from concurrent.futures import ThreadPoolExecutor as ThreadPool
        if num_process == 1:
            for item in object_list:
                self.func(item)
        else:
            pool = ThreadPool(max_workers=num_process)
            results = pool.map(self._unwrap_multi_process, object_list)
            pool.shutdown()
            return results

import threading

class RWlock(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._extra = threading.Lock()
        self.read_num = 0

    def read_acquire(self):
        with self._extra:
            self.read_num += 1
            if self.read_num == 1:
                self._lock.acquire()

    def read_release(self):
        with self._extra:
            self.read_num -= 1
            if self.read_num == 0:
                self._lock.release()

    def write_acquire(self):
        self._lock.acquire()

    def write_release(self):
        self._lock.release()


class RWLock(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.rcond = threading.Condition(self.lock)
        self.wcond = threading.Condition(self.lock)
        self.read_waiter = 0    # 等待获取读锁的线程数
        self.write_waiter = 0   # 等待获取写锁的线程数
        self.state = 0          # 正数：表示正在读操作的线程数   负数：表示正在写操作的线程数（最多-1）
        self.owners = []        # 正在操作的线程id集合
        self.write_first = True # 默认写优先，False表示读优先

    def write_acquire(self, blocking=True):
        # 获取写锁只有当
        me = threading.get_ident()
        with self.lock:
            while not self._write_acquire(me):
                if not blocking:
                    return False
                self.write_waiter += 1
                self.wcond.wait()
                self.write_waiter -= 1
        return True

    def _write_acquire(self, me):
        # 获取写锁只有当锁没人占用，或者当前线程已经占用
        if self.state == 0 or (self.state < 0 and me in self.owners):
            self.state -= 1
            self.owners.append(me)
            return True
        if self.state > 0 and me in self.owners:
            raise RuntimeError('cannot recursively wrlock a rdlocked lock')
        return False

    def read_acquire(self, blocking=True):
        me = threading.get_ident()
        with self.lock:
            while not self._read_acquire(me):
                if not blocking:
                    return False
                self.read_waiter += 1
                self.rcond.wait()
                self.read_waiter -= 1
        return True

    def _read_acquire(self, me):
        if self.state < 0:
            # 如果锁被写锁占用
            return False

        if not self.write_waiter:
            ok = True
        else:
            ok = me in self.owners
        if ok or not self.write_first:
            self.state += 1
            self.owners.append(me)
            return True
        return False

    def unlock(self):
        me = threading.get_ident()
        with self.lock:
            try:
                self.owners.remove(me)
            except ValueError:
                raise RuntimeError('cannot release un-acquired lock')

            if self.state > 0:
                self.state -= 1
            else:
                self.state += 1
            if not self.state:
                if self.write_waiter and self.write_first:   # 如果有写操作在等待（默认写优先）
                    self.wcond.notify()
                elif self.read_waiter:
                    self.rcond.notify_all()
                elif self.write_waiter:
                    self.wcond.notify()

    read_release = unlock
    write_release = unlock

if __name__ == '__main__':
    #  create ten reader threads
    for i in range(10):
        threading.Thread(target=calendar_reader, args=(i,)).start()
    #  ...but only two writer threads
    for i in range(2):
        threading.Thread(target=calender_writer, args=(i,)).start()
