class Canvas:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.shape = '.'
        self.matrix = [[self.shape] * self.n for _ in range(self.m)]
        self.layers = []
        self.layers_dict = {}
        return

    def reshape_matrix(self):
        if self.layers_dict and self.matrix:
            for elem in self.layers_dict.keys():
                for i in range(len(self.matrix)):
                    for j in range(len(self.matrix[0])):
                        layer = self.layers_dict[elem]
                        if layer.left_top[0] <= i <= layer.right_button[0] \
                                and layer.left_top[1] <= j <= layer.right_button[1]:
                            self.matrix[i][j] = layer.elem

    def print_matrix(self):
        for row in self.matrix:
            print(''.join(row))

    def add_rect(self, elem, size_x: int, size_y: int, pos: tuple):
        left_top = pos
        right_button = pos[0] + size_x - 1, pos[1] + size_y - 1
        rt = Rect(elem=elem, left_top=left_top, right_button=right_button)
        self.layers_dict[elem] = rt
        # self.layers.append(rt)
        return

    def move_rect(self, elem, pos: tuple):
        x, y = pos[0], pos[1]
        rect = self.layers_dict[elem]
        x_delta = x - rect.left_top[0]
        y_delta = y - rect.left_top[1]
        right_button_x = rect.right_button[0] + x_delta
        right_button_y = rect.right_button[1] + y_delta
        rect.left_top = (x, y)
        rect.right_button = (right_button_x, right_button_y)
        return


class Rect:
    def __init__(self, elem, left_top, right_button):
        self.elem = elem
        self.left_top = left_top
        self.right_button = right_button
        return


if __name__ == '__main__':
    can = Canvas(m=10, n=25)
    # can.print_matrix()
    can.add_rect(elem='a', size_x=4, size_y=6, pos=(0, 0))
    can.add_rect(elem='b', size_x=6, size_y=4, pos=(4, 21))
    can.add_rect(elem='c', size_x=5, size_y=5, pos=(2, 5))
    can.move_rect(elem='c', pos=(2, 7))
    can.reshape_matrix()
    can.print_matrix()
