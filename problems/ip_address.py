"""
示例 1：

输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]
示例 2：

输入：s = "0000"
输出：["0.0.0.0"]

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/restore-ip-addresses
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


def restoreIpAddresses(self, s: str) -> List[str]:
    SEG_COUNT = 4
    ans = list()
    segments = [0] * SEG_COUNT

    def dfs(segId: int, segStart: int):
        # 如果找到了 4 段 IP 地址并且遍历完了字符串，那么就是一种答案
        if segId == SEG_COUNT:
            if segStart == len(s):
                ipAddr = ".".join(str(seg) for seg in segments)
                ans.append(ipAddr)
            return

        # 如果还没有找到 4 段 IP 地址就已经遍历完了字符串，那么提前回溯
        if segStart == len(s):
            return

        # 由于不能有前导零，如果当前数字为 0，那么这一段 IP 地址只能为 0
        if s[segStart] == "0":
            segments[segId] = 0
            dfs(segId + 1, segStart + 1)

        # 一般情况，枚举每一种可能性并递归
        addr = 0
        for segEnd in range(segStart, len(s)):
            addr = addr * 10 + (ord(s[segEnd]) - ord("0"))
            if 0 < addr <= 0xFF:
                segments[segId] = addr
                dfs(segId + 1, segEnd + 1)
            else:
                break

    dfs(0, 0)
    return ans

"""
示例 1：

输入：queryIP = "172.16.254.1"
输出："IPv4"
解释：有效的 IPv4 地址，返回 "IPv4"
示例 2：

输入：queryIP = "2001:0db8:85a3:0:0:8A2E:0370:7334"
输出："IPv6"
解释：有效的 IPv6 地址，返回 "IPv6"

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/validate-ip-address
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""
def validIPAddress(self, queryIP: str) -> str:
    def isIPv4(ip: str) -> bool:
        return all(
            s and s.isdigit() and not (s[0] == '0' and len(s) > 1) and 0 <= int(s) <= 255 for s in sp) if len(
            sp := ip.split(".")) == 4 else False

    def isIPv6(ip: str) -> bool:
        return all(s and len(s) <= 4 and all(c in "0123456789ABCDEFabcdef" for c in s) for s in sp) if len(
            sp := ip.split(":")) == 8 else False

    if "." in queryIP and ":" not in queryIP and isIPv4(queryIP):
        return "IPv4"
    elif ":" in queryIP and "." not in queryIP and isIPv6(queryIP):
        return "IPv6"
    return "Neither"
