#!/usr/bin/env python3
"""prof_batch.py PORT MODEL CONTAINER MODE -- M concurrent short requests (M=1,2,4,8).
MODE=wall : per-request wall time -> per-step ms for each M (server uninstrumented)
MODE=regions : parse 'host region budget (M tokens' blocks from docker logs (GRIMOIRE_TIME_LAYER=all)"""
import json, re, statistics, subprocess, sys, threading, time, urllib.request
port, model, cont, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
P = ["Explain in two sentences why the sky is blue.", "Write a haiku about a lighthouse.",
     "List three prime numbers greater than 100.", "What is the capital of Australia? Answer briefly.",
     "Describe a sunset over the ocean in one paragraph.", "Give two tips for learning a new language.",
     "Summarize the plot of Hamlet in three sentences.", "Why do cats purr? Answer in two sentences."]
N = 64
def chat(i, out):
    body = json.dumps({"model": f"/models/{model}", "messages": [{"role": "user", "content": P[i]}],
                       "max_tokens": N}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(f'http://localhost:{port}/v1/chat/completions',
            data=body, headers={'Content-Type': 'application/json'}), timeout=900) as r:
        d = json.load(r)
    out[i] = (time.time() - t0, d['usage']['completion_tokens'])
for M in (1, 2, 4, 8):
    since = time.strftime('%Y-%m-%dT%H:%M:%S')
    time.sleep(1.1)
    out = [None] * M
    ts = [threading.Thread(target=chat, args=(i, out)) for i in range(M)]
    t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]; wall = time.time() - t0
    toks = sum(o[1] for o in out)
    if mode == 'wall':
        print(f'M={M}: {toks} tokens in {wall:.2f}s -> {toks/wall:.1f} tok/s total, '
              f'{wall/N*1000:.1f} ms per step (incl. {M} short prefills)', flush=True)
    else:
        log = subprocess.run(['docker', 'logs', '--since', since, cont], capture_output=True, text=True).stdout
        blocks = re.findall(r'host region budget \((\d+) tokens, all layers\):\n((?:      .*\n)+)', log)
        steps = [b for m, b in blocks if int(m) == M]
        agg = {}
        for b in steps:
            for name, ms in re.findall(r'      (.+?)\s+([\d.]+) ms', b):
                agg.setdefault(name.strip(), []).append(float(ms))
        print(f'M={M}: {len(steps)} decode steps profiled (median ms per step, synced regions):', flush=True)
        for k, v in sorted(agg.items(), key=lambda kv: -statistics.median(kv[1])):
            print(f'   {k:<28} {statistics.median(v):8.2f}')
