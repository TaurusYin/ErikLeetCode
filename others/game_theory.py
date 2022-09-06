
"""
从后往前推
你和你的朋友，两个人一起玩 Nim 游戏：
桌子上有一堆石头。
你们轮流进行自己的回合， 你作为先手 。
每一回合，轮到的人拿掉 1 - 3 块石头。
拿掉最后一块石头的人就是获胜者。
假设你们每一步都是最优解。请编写一个函数，来判断你是否可以在给定石头数量为 n 的情况下赢得游戏。如果可以赢，返回 true；否则，返回 false 。
链接：https://leetcode.cn/problems/nim-game
"""
from cmath import sqrt


def canWinNim(self, n: int) -> bool:
    return n % 4 != 0


"""
There are n bulbs that are initially off. You first turn on all the bulbs, then you turn off every second bulb.
首先，因为电灯一开始都是关闭的，所以某一盏灯最后如果是点亮的，必然要被按奇数次开关。
我们假设只有 6 盏灯，而且我们只看第 6 盏灯。需要进行 6 轮操作对吧，请问对于第 6 盏灯，会被按下几次开关呢？这不难得出，第 1 轮会被按，第 2 轮，第 3 轮，第 6 轮都会被按。
为什么第 1、2、3、6 轮会被按呢？因为 6=1*6=2*3。一般情况下，因子都是成对出现的，也就是说开关被按的次数一般是偶数次。但是有特殊情况，比如说总共有 16 盏灯，那么第 16 盏灯会被按几次?
16 = 1*16 = 2*8 = 4*4
其中因子 4 重复出现，所以第 16 盏灯会被按 5 次，奇数次。现在你应该理解这个问题为什么和平方根有关了吧？
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/bulb-switcher
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""
def bulbSwitch(self, n: int) -> int:
        return int(sqrt(n))