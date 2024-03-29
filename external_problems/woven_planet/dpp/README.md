## Enviroment
1) Prefer Linux OS/Mac OS
2) Prefer Python3.8+
3) requirement.txt: heapq/collections/wraps 

## QuickStart
Please prepare your target file first. The format is like
<unique record identifier><white_space><numeric value>
e.g.
1426828011 9
1426828028 350
1426828037 25
1426828056 231
1426828058 109
1426828066 111

You can also generate the target file by script(**./tools/mock_data_generator.py**)
```bash
python3 /Users/yineric/PycharmProjects/WovenPlanet/tools/mock_data_generator.py
```


This script is responsible for analyze the target file. Please follow the reference help doc below. (_**./tools/run_analyze.py**_)
> Usage: run_analyze.py [options]

Options:
  -h, --help            show this help message and exit
  -f TARGET_FILE, --file=TARGET_FILE
                        Input the target file path
  -x X_LARGEST, --x_largest=X_LARGEST
                        Give the X value for calculating X-largest item from
                        the target file
  -s SOLUTION, --solution=SOLUTION
                        Assign the solutions:
                        ["EntireSortSolution","MinHeapSolution", "MinHeapPartitionSolution" ], the default Solution is "MinHeapPartitionSolution"
> 

```bash
python3 /Users/yineric/PycharmProjects/WovenPlanet/tools/run_analyze.py -f /Users/yineric/PycharmProjects/WovenPlanet/resource/target_file -x 3 -s "MinHeapPartitionSolution"
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/29649471/1675343427619-105d9cfe-f314-4d8d-8559-af5b85ae5ad7.png#averageHue=%23313131&clientId=u8301f0a8-ac1b-4&from=paste&height=102&id=u0cd5b9e5&name=image.png&originHeight=203&originWidth=1808&originalType=binary&ratio=1&rotation=0&showTitle=false&size=42503&status=done&style=none&taskId=uedfc4ea7-70c8-47c8-bb94-bce0ee184a7&title=&width=904)



## Problem Description
```
Imagine a file in the following fixed format:
<unique record identifier><white_space><numeric value>
e.g.
1426828011 9
1426828028 350
1426828037 25
1426828056 231
1426828058 109
1426828066 111
.
.
.
Write a program that reads from 'stdin' the contents of a file, and optionally accepts
the absolute path of a file from the command line. The file/stdin stream is expected
to be in the above format. The output should be a list of the unique ids associated
with the X-largest values in the rightmost column, where X is specified by an input
parameter. For example, given the input data above and X=3, the following would be
valid output:
1426828028
1426828066
1426828056
Note that the output does not need to be in any particular order. Multiple instances
of the same numeric value count as distinct records of the total X. So if we have 4
records with values: 200, 200, 115, 110 and X=2 then the result must consist of the two
IDs that point to 200 and 200 and no more.
Your solution should take into account extremely large files.

What to return back to us
1. Your code in the language of your preference. If it’s in Python, Java or C++ it’s
preferable for us.
2. Include in your code comments about your solution's algorithmic complexity
for both time and memory.
3. Include instructions on how to build and run your code in a README file.
Please include operating system information if necessary.
4. Provide tests for your code to illustrate it works and it’s robust.
5. Please zip everything in a directory named your first.lastname/ and
return via email.
6. In your email response please let us know roughly how many hours you spent
for this exercise.
Notes
● For your reference, successful candidates reported to have spent
approximately 10 hours on average on this challenge.
● Write your project as if it will be reviewed by your peers and ultimately
deployed to production.
```
  
## Code Structure
/resource: target file and target_file partitions will be generated in this folder
/solution: conclude the main solutions in this "top__k_problems_"
/test: conclude test case
/tools: conclude scripts. 
          "mock_data_generate.py" is used to generate the mock data of target file.
          "run analyze" is used to start to analyze the target file
/utils: put the basic common class here. such as basic data structure, basic decorators, other basic common utils.

## Solution
 According to the requirement, the number of the total records is extremely huge, and it looks like the X of the "X-largest" number is not such huge since the sample X equals to 2. Otherwise, it does not require the particular order of the output. Here we figure out those solutions below. 
**"MinHeapPartitionSolution" which is default, should be best solution to handle with huge file.**

### MinHeapSolution 
a concrete subclass of BaseSolution that implements the Top-K computation using a min heap data structure. It adds an instance variable "min_heap" that holds the min heap, and the "analyze" function reads the target file, adds each row of data to the min heap, and computes the final Top-K results. 

The time complexity of this solution is O(N * log(K)) where N is the number of rows in the target file and K is the number of top records to be computed. The memory complexity of this solution is O(K) since the min heap data structure will store the top K smallest elements at any given time.

1. Construct a "min heap" with K size (K-largest) 
2. Read the target file row by row.
3. Read one row record, try to add the record into the Min Heap with K size. 
4. If the element value is higher than the lowest element of the heap, then pop the lowest element and add the current element into "heap".

Finally, the largest-K elements will be stored into this "min heap".

### MinHeapPartitionSolution 
Considering the size of the file is extremely huge. Maybe hundreds MB/GB/TB.
Spliting the data should be considered and it will accelerate the speed for the CPU insentive calculation by multi-process . 
Assume that we have a single standalone server to calculate the result with multiple process. 

Time complexity: O(N*logK / P) for each process, where N is the number of records, K is the value of X, and P is the number of processes since we are splitting the records into P chunks, processing each chunk in parallel, and each chunk has a time complexity of O(N/P * logK). The final total time complexity is O(N*logK).
Memory complexity: O(K + P), because we are storing the K largest values in the heap for each process, and P is the number of processes.
This solution will accelerate the process by multiprocess compared with the MinHeapSolution.
    

1. Make chunks to split the target file based on the size of the file.
2. One process handles with one chunk file and read the file row by row.
3. For each chunk file, each process will construct a "min-heap" similar with the **minHeapSolution **by read the target file row by row 
4. When the process finish reading the file & construct its own "min-heap", each process will return its "min-heap"
5. The main process get each process's "min-heap" and merge all the k-largest "min-heap" into the main "min-heap".


### EntireSortSolution  
This solution is the comparing method to verify the result. If the file is extremely huge, it will bring the OOM problem. Also, the performance is also not good comparing with MinHeapSolution.
The memory complexity of this code is O(n) and The time complexity of this code is O(nlogn) where n is the number of records in the target file

1. read the row into list
2. sort them and pop up the K-largest elements.

 
## Experiment
![image.png](https://cdn.nlark.com/yuque/0/2023/png/29649471/1675338498040-87003376-9fe4-4500-849f-c6bd0456ea95.png#averageHue=%2336393d&clientId=u8301f0a8-ac1b-4&from=paste&height=182&id=uaba540a3&name=image.png&originHeight=363&originWidth=320&originalType=binary&ratio=1&rotation=0&showTitle=false&size=42987&status=done&style=none&taskId=u5d4e1b5b-b8ad-436f-9b70-7525d593139&title=&width=160)
Analyze the 477 MB file and we can find that MinHeapPartitionSolution speed is fastest and EntireSortSolution is the slowest.
![image.png](https://cdn.nlark.com/yuque/0/2023/png/29649471/1675338448044-d752b071-050f-4ad4-8f85-557b1823c560.png#averageHue=%23303030&clientId=u8301f0a8-ac1b-4&from=paste&height=190&id=ucc5c11f3&name=image.png&originHeight=387&originWidth=1324&originalType=binary&ratio=1&rotation=0&showTitle=false&size=79172&status=done&style=none&taskId=u31a9655c-18eb-4872-8056-ed5c4817ef2&title=&width=651)

Test Case:
![image.png](https://cdn.nlark.com/yuque/0/2023/png/29649471/1675654459097-e1defb43-ea40-4135-aac3-b5b939a328ec.png#averageHue=%23666240&clientId=uf500a771-de17-4&from=paste&height=696&id=u11ff4cb8&name=image.png&originHeight=1392&originWidth=2666&originalType=binary&ratio=1&rotation=0&showTitle=false&size=468652&status=done&style=none&taskId=u3720c787-5ad5-421d-b683-7d729c5ba91&title=&width=1333)
