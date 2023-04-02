import os
import heapq
import traceback
from typing import List, Any
import concurrent.futures
from abc import ABC, abstractmethod
from dpp.utils.base_data_structure import MinHeap, ListItem
from dpp.utils.base_decorator import BaseTracer


def from_row_to_records(row) -> List:
    """
    Transform the row's value to float type for comparing the values.
    :param row:
    :return:
    """
    format_warning = "Please follow the correct format: <unique record identifier><white_space><numeric value>"
    try:
        row = row.replace('\n', '')
        timestamp, value = row.split(" ")
        if not timestamp or not value:
            raise format_warning
        return [timestamp, float(value)]
    except Exception:
        raise ValueError("Error: {}".format(format_warning))


class BaseSolution(ABC):
    def __init__(self, file_path: str = '../resource/target_file', k: int = 3):
        """
        :param file_path: input file path
        :param k: k value of the top-k
        """
        self.file_path = file_path
        self.k = k
        self.partition_files = []
        self.count_top_k = []

    @abstractmethod
    def analyze(self):
        """
        the function of calculating the Top-K result
        """
        pass

    def print_answer(self):
        """
        print the required answer through the 'self.count_top_k'(List[List])
        :return:
        """
        print('{} Method Result: '.format(self.__class__.__name__))
        for item in self.count_top_k:
            print(item[0])
        return


class MinHeapSolution(BaseSolution):
    """
    a concrete subclass of BaseSolution that implements the Top-K computation using a min heap data structure. It adds
    an instance variable "min_heap" that holds the min heap, and the "analyze" function reads the target file,
    adds each row of data to the min heap, and computes the final Top-K results.

    The time complexity of this solution is O(N * log(K)) where N is the number of rows in the target file
    and K is the number of top records to be computed.
    The memory complexity of this solution is O(K) since the min heap data structure will store the top K smallest
    elements at any given time.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_heap = MinHeap(k=self.k)

    @BaseTracer()
    def analyze(self) -> List[List]:
        min_heap = self.min_heap
        with open(self.file_path, 'r') as f:
            for row in f:
                list_item = from_row_to_records(row)
                min_heap.add_item(ListItem(list_item))
            f.close()
        for item in min_heap.heap:
            self.count_top_k.append(item.value)
        self.print_answer()
        return self.count_top_k


class MinHeapPartitionSolution(MinHeapSolution):
    """
    a subclass of MinHeapSolution that further implements parallel processing by splitting the target file into smaller
    chunks and computing the Top-K results for each chunk in separate processes. The "make_chunks" function splits the
    target file, and the "read_chunks" function reads each chunk, computes the Top-K results for that chunk and merges
    the results into the overall min heap.

    Time complexity: O(N*logK / P) for each process, where N is the number of records, K is the value of X, and P is
    the number of processes since we are splitting the records into P chunks, processing each chunk in parallel, and
    each chunk has a time complexity of O(N/P * logK). The final total time complexity is O(N*logK).

    Memory complexity: O(K + P), because we are storing the K largest values in the heap for each process, and P is the
    number of processes.
    This solution will accelerate the process by multiprocess compared with the MinHeapSolution.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.partition_files = []
        with open(self.file_path, 'r') as fp:
            for count, line in enumerate(fp):
                # enumerate(file_pointer) doesn’t load the entire file in memory,
                # so this is an efficient fasted way to count lines in a file.
                pass

        self.num_lines_of_target_file = count + 1
        self.num_process = 10
        self.chunk_size = int(self.num_lines_of_target_file / self.num_process)
        self.folder_path = None

    @BaseTracer()
    def make_chunks(self, chunk_size: int):
        """
        Split the target file into the smaller chunks
        :param chunk_size: the size of each chunk
        :return:
        """
        filename = self.file_path
        folder_path = '../resource/{}_partitions/'.format(filename)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        self.folder_path = folder_path
        reader = open(filename, 'rb')
        p = 0
        while True:
            try:
                lines = [next(reader) for _ in range(chunk_size)]
            except StopIteration as e:
                break
            if not lines:
                break
            p += 1
            f_name = '{}part_{}'.format(folder_path, str(p))
            self.partition_files.append(f_name)
            file_object = open(f_name, 'wb')
            for line in lines:
                file_object.write(line)
            file_object.close()
        reader.close()
        return

    def read_chunks(self, partition_file: str):
        """
        Handle with the chunks one by one. Construct the partition_heap for each chunk in multiprocess,
        then merge all the elements of each partition_heap into the total heap(self.min_heap)
        :param partition_file:
        :return:
        """
        partition_heap = MinHeap(k=self.k)
        with open(partition_file, 'r') as f:
            for row in f:
                list_item = from_row_to_records(row)
                partition_heap.add_item(ListItem(list_item))
            f.close()
        '''
        for _ in range(len(partition_heap.heap)):
            HEAP.add_item(ListItem(heapq.heappop(partition_heap.heap).value))
        '''
        return partition_heap

    def clear_chunks(self):
        for file_path in self.partition_files:
            os.remove(file_path)

    @BaseTracer()
    def analyze(self) -> List[List]:
        """
        1. make chunks: split large target file into smaller chunks
        2. read chunks: calculate the top-k for each chunk through multiprocess
        3. merge each chunks' heap into the entire heap (self.min_heap).
        :return:
        """
        if self.chunk_size > 0:
            self.make_chunks(self.chunk_size)
        elif self.chunk_size == 0:
            self.partition_files.append(self.file_path)
            self.num_process = 1
        if self.partition_files:
            # Parallel Execution of the files
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_process) as executor:
                partition_heap_containers = executor.map(self.read_chunks, self.partition_files)
        for partition_heap in partition_heap_containers:
            for _ in range(len(partition_heap.heap)):
                heap_min_item = heapq.heappop(partition_heap.heap).value
                self.min_heap.add_item(ListItem(heap_min_item))
        for item in self.min_heap.heap:
            self.count_top_k.append(item.value)
        self.print_answer()
        if self.chunk_size > 0:
            self.clear_chunks()
        return self.count_top_k


class EntireSortSolution(BaseSolution):
    """
    Read the target file into memory and sort all the list.
    The memory complexity of this code is O(n) and The time complexity of this code is O(nlogn)
    where n is the number of records in the target file

    Note: this is not the final solution since it is just a comparing solution.
          if the target file is extremely huge, it will come up with such "OOM" problem
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @BaseTracer()
    def analyze(self) -> List[List]:
        target_list = []
        with open(self.file_path, 'r') as f:
            for row in f:
                list_item = from_row_to_records(row)
                target_list.append(list_item)
            f.close()
        count_all = sorted(target_list, key=lambda x: x[1], reverse=True)
        self.count_top_k = count_all[:self.k]
        self.print_answer()
        return self.count_top_k


if __name__ == '__main__':
    MinHeapPartitionSolution().analyze()
    MinHeapSolution().analyze()
    EntireSortSolution().analyze()
