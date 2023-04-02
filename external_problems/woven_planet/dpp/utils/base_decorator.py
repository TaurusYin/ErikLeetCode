import time
import traceback
from functools import wraps


class BaseTracer(object):
    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            func_name = func.__name__
            params = "args:{} kwargs:{}".format(args, kwargs)
            log_str = '[INFO] {} function is called, Params: ({})'.format(func_name, params)
            start_time = time.time()
            res = None
            try:
                res = func(*args, **kwargs)
            except FileNotFoundError as fe:
                error_message = traceback.format_exc()
                print("{}, FileNotFoundError: {}".format(log_str, error_message))
            except Exception as e:
                error_message = traceback.format_exc()
                print("{}, Error: {}".format(log_str, error_message))
            end_time = time.time()
            tolerance = 100000
            elapsed_time = float(int((end_time - start_time) * tolerance) / tolerance)
            print('{}, Elapsed time : {} seconds'.format(log_str, str(elapsed_time)))
            return res

        return wrapped_function
