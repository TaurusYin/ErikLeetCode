from datetime import datetime


class TimeUtils:
    def __init__(self):
        return

    @staticmethod
    def to_timestamp(input, unit='s') -> int:
        import time
        import pytz
        import datetime
        if isinstance(input, str):
            if 'T' in input and 'Z' in input:
                local_tz = pytz.timezone('Asia/Chongqing')
                local_format = "%Y-%m-%d %H:%M"
                utc_format = '%Y-%m-%dT%H:%M:%SZ'
                utc_dt = datetime.datetime.strptime(input, utc_format)
                local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
                time_str = local_dt.strftime(local_format)
                if unit == 's':
                    return int(time.mktime(time.strptime(time_str, local_format)))
                elif unit == 'ms':
                    return int(time.mktime(time.strptime(time_str, local_format))) * 1000
            if unit == 's':
                return int(time.mktime(time.strptime(input, "%Y-%m-%d %H:%M:%S")))
            elif unit == 'ms':
                return int(time.mktime(time.strptime(input, "%Y-%m-%d %H:%M:%S"))) * 1000
        if isinstance(input, datetime.datetime):
            dt_str = input.strftime("%Y-%m-%d %H:%M:%S")
            if unit == 's':
                return int(time.mktime(time.strptime(dt_str, "%Y-%m-%d %H:%M:%S")))
            elif unit == 'ms':
                return int(time.mktime(time.strptime(dt_str, "%Y-%m-%d %H:%M:%S")) * 1000)
        if isinstance(input, int):
            digit = len(str(input))
            if digit == 13 and unit == 's':
                input = int(input / 1000)
            elif digit == 10 and unit == 'ms':
                input = int(input * 1000)
            return input
        if isinstance(input, np.datetime64):
            input = input.astype('datetime64[s]').astype('int')
            return input

    @staticmethod
    def to_datetime(input) -> datetime:
        digit = len(str(input))
        if digit == 13:
            input = int(input / 1000)

        output = input
        if isinstance(input, str):
            output = datetime.datetime.strptime(input, "%Y-%m-%d %H:%M:%S")
        if isinstance(input, int):
            dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(input))
            output = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        if isinstance(input, datetime.datetime):
            output = input
        return output

    @staticmethod
    def to_time_str(input, unit='s') -> str:
        output = input
        if isinstance(input, datetime.datetime):
            output = input.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(input, int):
            digit = len(str(input))
            if digit == 13:
                input = int(input / 1000)
            elif digit == 10:
                input = int(input)
            output = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(input))
        return output
