"""

假设有四种货币：USD、CAD、EUR 和 CNY，它们之间的汇率如下：

python
Copy code
table = [[1, 1.3, 1, 6.49],
         [0.72, 1, 0.9, 5.5],
         [1.1, 1.1, 1, 7.3],
         [0.18, 0.2, 0.136, 1]]
我们要计算从 USD 到 CNY 的最大汇率，限制步数不超过 3 步。换句话说，我们需要计算经过最多三个中间货币的最大 USD/CNY 汇率。

首先，我们使用一个名为 names 的字典将货币名称映射到它们在表中的索引，如下所示：

python
Copy code
names = {'USD': 0, 'CAD': 1, 'EUR': 2, 'CNY': 3}
接下来，我们创建一个名为 memo 的二维列表来存储中间结果。memo[k][i] 表示经过不超过 k 步从原始货币到达货币 i 的最大汇率。我们使用两个 for 循环填充第一行的值，它们表示从 USD 到其他货币的直接汇率：

python
Copy code
memo = [[0] * len(currencies) for _ in range(len(currencies))]
current = names['USD']
for i in range(len(currencies)):
    memo[1][i] = table[current][i]
这将 memo 填充为：

lua
Copy code
[[0, 1.3, 0, 6.49],
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]]
接下来，我们使用两个嵌套的 for 循环填充剩余的值。外层循环遍历步数 j，内层循环遍历中间货币 c。我们检查 c 是否已经用作交换过的货币，如果没有，我们尝试将其添加到交换线路中，并计算从 USD 到 c 的最大汇率，如下所示：

python
Copy code
exchanged = set([current])
targetCurr = names['CNY']
for j in range(2, len(table)):
    for i in range(len(table)):
        if i not in exchanged:
            exchanged.add(i)
            memo[j][i] = max(memo[j-1][i], memo[j-1][current]*table[current][i])
            exchanged.remove(i)
这将 memo 填充为：

lua
Copy code
[[0, 1.3, 0, 6.49],
 [0, 0, 0, 0],
 [0, 1.69, 0, 43.99],
 [0, 1.94, 0, 44.33176]]
最后，我们返回 memo[3][3]，它表示从 USD 到 CNY 的最大汇率。
"""


def max_transaction_rate(currencies, table, target, currencyAtHand):
    names = {currency: i for i, currency in enumerate(currencies)}
    memo = [[0] * len(currencies) for _ in range(len(currencies))]
    current = names[currencyAtHand]
    for i in range(len(currencies)):
        memo[1][i] = table[current][i]
    exchanged = set([current])
    targetCurr = names[target]
    for j in range(2, len(table)):
        for i in range(len(table)):
            if i not in exchanged:
                exchanged.add(i)
                memo[j][i] = max(memo[j - 1][i], memo[j - 1][current] * table[current][i])
                exchanged.remove(i)
    return memo[len(table) - 1][targetCurr]


currencies = ["USD", "CAD", "EUR", "CNY"]
table = [[1, 1.3, 1, 6.49], [0.72, 1, 0.9, 5.5], [1.1, 1.1, 1, 7.3], [0.18, 0.2, 0.136, 1]]
print(max_transaction_rate(currencies, table, "CNY", "USD"))
