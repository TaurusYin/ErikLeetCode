import os
import sys
import importlib
from optparse import OptionParser

project_dir = os.path.dirname(sys.path[0])
sys.path.append(project_dir)
from dpp.solution import EntireSortSolution, MinHeapSolution, MinHeapPartitionSolution

parser = OptionParser()
parser.add_option("-f", "--file", dest="target_file",
                  help='Input the target file path', default='../resource/target_file')
parser.add_option("-x", "--x_largest", dest="x_largest", default=3, type=int,
                  help='Give the X value for calculating X-largest item from the target file')
parser.add_option("-s", "--solution", dest="solution", default="MinHeapPartitionSolution",
                  help='Assign the solutions: ["EntireSortSolution","MinHeapSolution", "MinHeapPartitionSolution"]')

if __name__ == '__main__':
    (options, args) = parser.parse_args()
    target_file = options.target_file
    x_largest = options.x_largest
    method = options.solution
    # Load "module.submodule.MyClass"
    SolutionClass = getattr(importlib.import_module("solution.top_k_problems"), method)
    # Instantiate the class (pass arguments to the constructor, if needed)
    solution = SolutionClass(file_path=target_file, k=x_largest)
    solution.analyze()
