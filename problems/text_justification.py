from typing import List


def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
    rtv = []
    line, line_chars = [], 0
    for word in words:
        if line_chars + len(word) + len(line) > maxWidth:
            # 字符数 + 空格数 > 最大长度
            spaces = maxWidth - line_chars
            if len(line) == 1:
                rtv.append(line[0] + ' ' * max(maxWidth - line_chars, 0))
            else:
                x, y = divmod(spaces, len(line) - 1)
                line = [wd + ' ' if idx < y else wd for idx, wd in enumerate(line)]
                rtv.append((' ' * x).join(line))
            line, line_chars = [word], len(word)
        else:
            line.append(word)
            line_chars += len(word)
    rtv.append(' '.join(line) + ' ' * (maxWidth - line_chars - len(line) + 1))
    return rtv
