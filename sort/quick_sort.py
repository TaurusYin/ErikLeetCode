def quicksort(array):
    if len(array) < 2:
        return array
    else:
        pivot_index = 0
        pivot = array[pivot_index]
        less_part = [i for i in array[pivot_index + 1:] if i <= pivot]
        greater_part = [i for i in array[pivot_index + 1:] if i > pivot]
        return quicksort(less_part) + [pivot] + quicksort(greater_part)


if __name__ == '__main__':
    print(quicksort([100, 2, 3, 4, 10, 40]))
