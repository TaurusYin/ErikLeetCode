# envelopes = sorted(envelopes, key=lambda x: (x[0], -x[1]))


arr = [[1, 2, 3], [3, 2, 1], [4, 2, 1], [6, 4, 3]]
indices = [[2, 0], [0, 1]]
def custom_sort(x):
    tmp_sort_method = []
    print(x)
    for indice in indices:
        idx = indice[0]
        if indice[1] == 1:
            res = -x[idx]
        else:
            res = x[idx]
        tmp_sort_method.append(res)
    return tmp_sort_method
arr.sort(key=lambda x: custom_sort(x))

print()


def compare(a, b):
    if a[2] > b[2]:
        return 1
    elif a[2] < b[2]:
        return -1
    else:
        return 0
    return


x = arr.sort(key=functools.cmp_to_key(compare))
print(x)
