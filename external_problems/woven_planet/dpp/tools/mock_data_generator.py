import random
import os
import sys

project_dir = os.path.dirname(sys.path[0])
sys.path.append(project_dir)

from dpp.utils.base_decorator import BaseTracer


class MockData:
    def __init__(self):
        self.num_record = 100000 * 200
        self.timestamp_range = (1426828011, 1526828011)
        self.value_range = (1, 10000000 * 100000)
        self.file_path = '../dpp/resource/target_file'

    def generate_one_record(self):
        random_timestamp = random.randint(self.timestamp_range[0], self.timestamp_range[1])
        random_value = random.randint(self.value_range[0], self.value_range[1])
        record_string = '{} {}'.format(random_timestamp, random_value)
        return record_string

    @BaseTracer()
    def generate_target_file(self):
        try:
            with open(self.file_path, 'w') as f:
                for i in range(self.num_record):
                    record = self.generate_one_record()
                    print(record)
                    f.write('{}\n'.format(record))
        except Exception as e:
            print(e)
        pass


if __name__ == '__main__':
    md = MockData()
    print("generating files... It will take around 15 mins to generate the 500MB target file")
    md.generate_target_file()
    print("solution analyze... each solution will take around 30s ~ 100s")
