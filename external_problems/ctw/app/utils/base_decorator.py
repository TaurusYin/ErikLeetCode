from functools import wraps
import logging
import time, traceback, sys


class CommonLogger(object):
    def __init__(self, method='print'):
        self.method = method

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            method = self.method

            def print_log(info):
                if method == 'print':
                    print(info)
                elif method == 'fc':
                    logging.info(info)

            func_name = func.__name__
            # args_res = list(map(lambda x: str(x), args))
            # params = ','.join(list(args_res))
            params = str(kwargs)[:200]
            if self.method == 'print':
                log_str = '[INFO] {} function is called, Params: ({})'.format(func_name, params)
            elif self.method == 'fc':
                log_str = '{} function is called, Params: ({})'.format(func_name, params)
            start_time = time.time()
            res = None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                import traceback
                error_message = traceback.format_exc()
                print_log("{}, Error: {}".format(log_str, error_message))
            end_time = time.time()
            elapsed_time = round((end_time - start_time), 6)
            print_log('{}, Elapsed time : {} seconds'.format(log_str, str(elapsed_time)))
            return res

        return wrapped_function
