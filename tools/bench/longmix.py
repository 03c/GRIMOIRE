#!/usr/bin/env python3
"""longmix.py PORT MODEL -- 8 concurrent greedy chats with very different prompt lengths
(~50 .. ~3000 tokens), 32 new tokens each: batched rows with different split-K counts."""
import json, sys, threading, urllib.request
port, model = int(sys.argv[1]), sys.argv[2]
para = ("The history of lighthouses spans more than two thousand years, from the Pharos of "
        "Alexandria to automated LED beacons. Keepers trimmed wicks, wound clockwork, and logged "
        "the weather through long winter nights. ")
reps = [1, 8, 20, 40, 60, 80, 100, 120]
out = {}
def chat(i):
    msg = para * reps[i] + f"\n\nQuestion {i}: summarize the text above in one sentence."
    body = json.dumps({"model": f"/models/{model}", "messages": [{"role": "user", "content": msg}],
                       "max_tokens": 32, "temperature": 0}).encode()
    with urllib.request.urlopen(urllib.request.Request(f'http://localhost:{port}/v1/chat/completions',
            data=body, headers={'Content-Type': 'application/json'}), timeout=900) as r:
        d = json.load(r)
    out[i] = (d['usage']['prompt_tokens'], d['choices'][0]['message'].get('content') or '')
ts = [threading.Thread(target=chat, args=(i,)) for i in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
for k in sorted(out): print(k, out[k][0], repr(out[k][1][:80]))
