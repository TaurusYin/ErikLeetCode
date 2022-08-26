

# https://labuladong.github.io/algo/2/20/27/
from collections import OrderedDict


def sliding_windows(s):
    windows, needs = OrderedDict(), OrderedDict()
    left = right = 0
    for right in range(len(s)):
        right_elem = s[right]
        print(right_elem)
        windows.update()
        while "windows needs shrink":
            left_elem = s[left]
            left += 1
            windows.update()
"""
void slidingWindow(string s) {
    unordered_map<char, int> window;
    
    int left = 0, right = 0;
    while (right < s.size()) {
        // c 是将移入窗口的字符
        char c = s[right];
        // 增大窗口
        right++;
        // 进行窗口内数据的一系列更新
        ...

        /*** debug 输出的位置 ***/
        printf("window: [%d, %d)\n", left, right);
        /********************/
        
        // 判断左侧窗口是否要收缩
        while (window needs shrink) {
            // d 是将移出窗口的字符
            char d = s[left];
            // 缩小窗口
            left++;
            // 进行窗口内数据的一系列更新
            ...
        }
    }
}
"""