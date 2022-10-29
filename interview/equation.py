from collections import defaultdict
import re

input = ['a=b+1', 'b=c+2', 'c=2']
input = ['a=b+4', 'b=c+d', 'd=4', 'c=3+2']

equation_dict = defaultdict()
final_dict = {}
for item in input:
    k, v = item.split('=')
    equation_dict[k] = v
    final_dict[k] = v

while final_dict != {}:
    for k, v in equation_dict.items():
        if final_dict == {}:
            break
        if v.isdigit():
            final_dict.pop(k)
            continue
        t = list(v)
        for i in range(len(v)):
            if v[i] in equation_dict:
                t[i] = equation_dict[v[i]]
        t_str = ''.join(t)
        if not bool(re.search('[a-z]', t_str)):
            t_str = str(eval(t_str))
        equation_dict[k] = t_str
        final_dict[k] = t_str
        if t_str.isdigit():
            final_dict.pop(k)
        print('e:{}'.format(equation_dict))
        print('f:{}'.format(final_dict))
