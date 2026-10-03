#!/usr/bin/env python3
"""dprof_table.py LOG -- GRIMOIRE_PROFILE_PREFILL device-time blocks -> median ms per region for M=1,2,4,8."""
import re, statistics, sys
log = open(sys.argv[1]).read()
blocks = re.findall(r"device prefill breakdown \((\d+) tokens\):\n((?:      .*\n)+)", log)
Ms = (1, 2, 4, 8)
res = {}
for M in Ms:
    steps = [b for m, b in blocks if int(m) == M]
    agg = {}
    for b in steps:
        for name, ms in re.findall(r"      (.+?)\s+([\d.]+) ms", b):
            agg.setdefault(name.strip(), []).append(float(ms))
    res[M] = ({k: statistics.median(v) for k, v in agg.items()}, len(steps))
names = sorted({k for M in Ms for k in res[M][0]}, key=lambda k: -res[8][0].get(k, 0))
print("region (device ms/step)".ljust(30) + "".join(("M=%d" % M).rjust(9) for M in Ms))
for k in names:
    print(k.ljust(30) + "".join(("%.3f" % res[M][0].get(k, 0)).rjust(9) for M in Ms))
print("TOTAL".ljust(30) + "".join(("%.2f" % sum(res[M][0].values())).rjust(9) for M in Ms))
print("steps".ljust(30) + "".join(str(res[M][1]).rjust(9) for M in Ms))
