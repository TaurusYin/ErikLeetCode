import time, copy, hashlib, logging
from collections import OrderedDict
from datetime import datetime
from functools import wraps


class CommonLogger(object):
    def __init__(self, method='print'):
        self.method = method
        self.output = []

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            method = self.method

            def print_log(info):
                if method == 'print':
                    print(info)
                elif method == 'fc':
                    logging.info(info)
                return info
            func_name = func.__name__
            func_args = copy.deepcopy(args)
            params = func.__code__.co_varnames
            params_dict = dict(zip(params, func_args))
            m = hashlib.md5()
            m.update((func_name + str(func_args)).encode('utf-8'))
            func_uid = m.hexdigest()[:5]
            call_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            params = str(kwargs)[:200]
            if self.method == 'print':
                log_str = '[INFO] {} function is called, Params: ({})'.format(func_name, params)
                log_str = '[INFO] {}, func name: {}, id:{}, locals: {}'.format(call_time,
                                                                               func.__name__,
                                                                               func_uid,
                                                                               params_dict)
                record = {
                    'call_time': call_time,
                    'func_name': func.__name__,
                    'func_uid': func_uid,
                    'params': params_dict
                }
                self.output.append(record)

            elif self.method == 'fc':
                log_str = '{} function is called, Params: ({})'.format(func_name, params)
                log_str = '{}, func name: {}, id:{}, locals: {}'.format(call_time,
                                                                        func.__name__,
                                                                        func_uid,
                                                                        params_dict)
            start_time = time.time()
            res = None
            info_str = print_log('\033[0;31;40m{}\033[0m'.format(log_str))
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                import traceback
                error_message = traceback.format_exc()
                print_log("{}, Error: {}".format(log_str, error_message))
            end_time = time.time()
            elapsed_time = round((end_time - start_time), 6)
            print_log('\033[0;32;40m{}, return: {}, Elapsed time: {} seconds\033[0m'.format(log_str, res, str(elapsed_time)))
            return res

        return wrapped_function


class CommonDecorator:
    def __init__(self):
        return

    def show_call_time(self, func):
        def wrap(*args):
            func_name = func.__name__
            func_args = copy.deepcopy(args)
            params = func.__code__.co_varnames
            params_dict = dict(zip(params, func_args))
            m = hashlib.md5()
            m.update((func_name + str(func_args)).encode('utf-8'))
            func_uid = m.hexdigest()[:5]
            call_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print('\033[0;31;40m{}, func name: {}, id:{}, locals: {}\033[0m'.format(call_time, func.__name__, func_uid,
                                                                                    params_dict))
            res = func(*args)
            print(
                '\033[0;32;40m{}, func name: {}, id:{}, locals: {}, return: {}\033[0m'.format(call_time, func.__name__,
                                                                                              func_uid, params_dict,
                                                                                              res))

        return wrap

    def __call__(self, *args, **kwargs):
        def foo(func):
            return self.show_call_time(func)

        return foo


# decorator = CommonDecorator()

@CommonLogger()
def f1(a, b):
    print('result:{} {}'.format(a, b))
    return a + b


if __name__ == '__main__':
    print()
    f1(1, 2)
    f1(1, 3)
