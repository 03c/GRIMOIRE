#!/usr/bin/env python3
# tool_e2e.py PORT [quick] -- OpenAI tool calling and reasoning split through grimoire-server.
import json, sys, urllib.request
port = sys.argv[1]; url = "http://127.0.0.1:%s/v1/chat/completions" % port
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]
def post(body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:300]}
def show(tag, r):
    if "http_error" in r: print(tag, "HTTP", r["http_error"], r["body"]); return None
    c = r["choices"][0]; m = c["message"]
    print("%s finish=%s tool_calls=%s" % (tag, c.get("finish_reason"), json.dumps(m.get("tool_calls"))[:300]))
    print("   reasoning: %r" % (m.get("reasoning_content") or "")[-160:])
    print("   content:   %r" % (m.get("content") or "")[-200:])
    return m
msgs = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
m = show("1. tool request:", post({"model": "m", "messages": msgs, "tools": TOOLS, "max_tokens": 600, "temperature": 0}))
if m and m.get("tool_calls"):
    tc = m["tool_calls"][0]
    msgs2 = msgs + [{"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"]},
                    {"role": "tool", "tool_call_id": tc.get("id", "call_0"), "content": json.dumps({"city": "Paris", "temperature_c": 18, "sky": "light rain"})}]
    show("2. with tool result:", post({"model": "m", "messages": msgs2, "tools": TOOLS, "max_tokens": 600, "temperature": 0}))
# streaming tool call
body = {"model": "m", "messages": msgs, "tools": TOOLS, "max_tokens": 600, "temperature": 0, "stream": True}
req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
deltas = []; finish = None
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:]); ch = (d.get("choices") or [{}])[0]
            if ch.get("delta", {}).get("tool_calls"): deltas.append(ch["delta"]["tool_calls"])
            if ch.get("finish_reason"): finish = ch["finish_reason"]
    print("3. streamed: finish=%s tool_call deltas=%s" % (finish, json.dumps(deltas)[:300]))
except urllib.error.HTTPError as e:
    print("3. streamed: HTTP", e.code, e.read().decode()[:200])
show("4. no tools:", post({"model": "m", "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}], "max_tokens": 500, "temperature": 0}))
