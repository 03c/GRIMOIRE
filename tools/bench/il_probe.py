#!/usr/bin/env python3
# il_probe.py PORT -- user A streams a long answer; when A has 40 tokens, user B sends a ~4K-token
# prompt.  Prints A's longest gap between tokens, A's tokens while B was prefilling, B's TTFT and text.
import json, sys, threading, time, urllib.request
port = sys.argv[1]; url = "http://127.0.0.1:%s/v1/chat/completions" % port
book = open("/mnt/storage/isos/grimoire-runs/bench-1003/book/1661-0.txt", encoding="utf-8", errors="ignore").read()
long_text = book[20000:20000 + 15000]
def stream(content, n, rec):
    body = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": 0.0, "stream": True}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    rec["t0"] = time.time(); rec["times"] = []; rec["text"] = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:]); ch = d.get("choices") or [{}]
            delta = ch[0].get("delta", {})
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                rec["times"].append(time.time()); rec["text"] += piece
A, B = {}, {}
ta = threading.Thread(target=stream, args=("Write a very long, detailed story about a lighthouse keeper and a storm.", 700, A))
ta.start()
while len(A.get("times", [])) < 40: time.sleep(0.01)
tb = threading.Thread(target=stream, args=(long_text + "\n\nIn one sentence: who are the main characters of the text above?", 48, B))
tb.start(); tb.join(); ta.join()
gaps = [b - a for a, b in zip(A["times"], A["times"][1:])]
b_first = B["times"][0] if B["times"] else None
during = sum(1 for t in A["times"] if B["t0"] <= t <= (b_first or B["t0"]))
print("A: %d chunks, longest gap %.2f s; chunks while B prefilled: %d" % (len(A["times"]), max(gaps), during))
print("B: TTFT %.2f s, %d chunks, text: %r" % ((b_first or 0) - B["t0"], len(B["times"]), B["text"][-140:]))
