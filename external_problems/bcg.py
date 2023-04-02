"""
Problem:
String. isValid. isinValid {}

‘( a ) ‘ <- valid
‘(a’ <- invalid

‘())’ <- invalid
‘()’ <- valid

‘ [[ a ]] ‘ <- valid
‘[‘ <- valid
']' <- valid
‘[[‘ <- invalid
']]' -> invalid

"""
def is_valid(target_str: str) -> bool:
    pairs = {
        ")": "(",
        "}": "{",
        "]": "[",
    }
    source_elems = [")", "(", "}", "{", "]", "["]

    stack = []
    for elem in target_str:
        if elem in pairs:
            if not stack or stack[-1] != pairs[elem]:
                return False
            stack.pop()
        elif elem in source_elems:
            stack.append(elem)
    print(stack)
    return not stack


def is_valid_follow_up(target_str: str) -> bool:
    pairs = {
        "]": "[",
    }
    source_elems = ["[", "]"]
    invalid_elems = ["[[", "]]"]
    stack = []
    for elem in target_str:
        if elem in pairs:
            if not stack or stack[-1] != pairs[elem]:
                return False
            stack.pop()
        elif elem in source_elems:
            stack.append(elem)
    print(stack)
    remain_str = ''.join(stack)
    if remain_str in invalid_elems:
        return False
    else:
        return True


if __name__ == '__main__':
    target_str = '[[a]]'
    target_str = '['
    target_str = '[['
    target_str = ']]'

    # target_str = '( (a) )'
    print(is_valid_new(target_str))
