import unittest
from dpp.solution.top_k_problems import EntireSortSolution
from dpp.solution.top_k_problems import MinHeapSolution
from dpp.solution.top_k_problems import MinHeapPartitionSolution
from dpp.tools.mock_data_generator import MockData


class Test(unittest.TestCase):
    def test_analyze_small_file_for_each_solution(self):
        target_file = '../resource/example'
        answer = set(['1426828066', '1426828028', '1426828056'])
        entire_sort_solution_res = set(map(lambda x: x[0], EntireSortSolution(file_path=target_file).analyze()))
        min_heap_solution_res = set(map(lambda x: x[0], MinHeapSolution(file_path=target_file).analyze()))
        min_heap_partition_solution_res = set(
            map(lambda x: x[0], MinHeapPartitionSolution(file_path=target_file).analyze()))
        assert set(entire_sort_solution_res) == set(min_heap_solution_res) == set(
            min_heap_partition_solution_res) == answer
        pass

    def test_analyze_small_file_for_test_case_1(self):
        target_file = '../resource/test_case_1'
        answer = set(['1426828011', '1426828028', '1426828037'])
        entire_sort_solution_res = set(map(lambda x: x[0], EntireSortSolution(file_path=target_file).analyze()))
        min_heap_solution_res = set(map(lambda x: x[0], MinHeapSolution(file_path=target_file).analyze()))
        min_heap_partition_solution_res = set(
            map(lambda x: x[0], MinHeapPartitionSolution(file_path=target_file).analyze()))
        assert set(entire_sort_solution_res) == set(min_heap_solution_res) == set(
            min_heap_partition_solution_res) == answer
        pass

    def test_analyze_small_file_for_test_case_2(self):
        target_file = '../resource/test_case_2'
        answer = set(['1426828011'])
        entire_sort_solution_res = set(map(lambda x: x[0], EntireSortSolution(file_path=target_file).analyze()))
        min_heap_solution_res = set(map(lambda x: x[0], MinHeapSolution(file_path=target_file).analyze()))
        min_heap_partition_solution_res = set(
            map(lambda x: x[0], MinHeapPartitionSolution(file_path=target_file).analyze()))
        assert set(entire_sort_solution_res) == set(min_heap_solution_res) == set(
            min_heap_partition_solution_res) == answer
        pass

    def test_analyze_small_file_for_file_not_found(self):
        target_file = '../dpp/resource/test_case_x'
        try:
            min_heap_partition_solution_res = set(
            map(lambda x: x[0], MinHeapPartitionSolution(file_path=target_file).analyze()))
        except FileNotFoundError as fe:
            assert True
        pass

    def test_analyze_small_file_for_file_format_error(self):
        target_file = '../resource/test_case_format_error'
        MinHeapPartitionSolution(file_path=target_file).analyze()
        pass



