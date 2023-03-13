
"""
多个字符串数组:
s1 = ['a1', 'a22', 'a3', 'a44', 'a55' ..]
s2 = ['b1, 'b2', 'b33', 'b4', 'b55' ..]
s3 = ['c1, 'c2', 'c33', 'c44', 'c55' ..]

sn = ....
       s2 s1 s3  s1 s3
str = 'b4 a3 c1 a55 c55'

new_str = 'b4 a3 c1 a55 c55'
"""

str =     'b4 a3 c1 a55 c55 c33'
new_str = 'b4 a3 c1 a55 c55 c44'


def isEqual(str, new_str):
    s1 = ['a1', 'a22', 'a3', 'a44', 'a55']
    s2 = ['b1', 'b2', 'b33', 'b4', 'b55']
    s3 = ['c1', 'c2', 'c33', 'c44', 'c55']
    sum_hash = {}
    for e in s1: sum_hash[e] = 's1'
    for e in s2: sum_hash[e] = 's2'
    for e in s3: sum_hash[e] = 's3'

    str_list = str.split(' ')
    new_str_list = new_str.split(' ')
    if len(str_list) != len(new_str_list):
        return False
    for i in range(0, len(str_list)):
        if new_str_list[i] not in sum_hash:
            return False
        if new_str_list[i] in sum_hash and str_list[i] in sum_hash:
            if sum_hash[new_str_list[i]] == sum_hash[str_list[i]]:
                continue
            else:
                return False
    return True
res = isEqual(str, new_str)
print(res)






