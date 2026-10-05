# render_jinja.py TEMPLATE CASES -- transformers' chat-template semantics (jinja2 sandbox,
# trim/lstrip blocks, tojson = json.dumps(ensure_ascii=False)); OpenAI tool_calls arguments
# (JSON strings) become dicts, as servers do before applying the template.
import json, sys
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.ext import loopcontrols
def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
def raise_exception(m): raise Exception(m)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
tmpl = env.from_string(open(sys.argv[1]).read())
for case in json.load(open(sys.argv[2])):
    msgs = []
    for m in case["messages"]:
        m = dict(m)
        if "tool_calls" in m:
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                a = tc["function"].get("arguments")
                if isinstance(a, str): tc["function"]["arguments"] = json.loads(a)
                tcs.append(tc)
            m["tool_calls"] = tcs
        msgs.append(m)
    kw = dict(case.get("chat_template_kwargs", {}))
    if "reasoning_effort" in case: kw["reasoning_effort"] = case["reasoning_effort"]
    try:
        out = tmpl.render(messages=msgs, tools=case.get("tools"), add_generation_prompt=True, **kw)
    except Exception as e:
        out = "ERROR: " + str(e)
    sys.stdout.write(out + "\n=====CASE=====\n")
