sort_res = sorted(stats.items(), key=lambda x: x[1], reverse=True)
envelopes = sorted(envelopes, key=lambda x: (x[0], -x[1]))
intervals = sorted(intervals, key=lambda x: (x[0], -x[1]))


