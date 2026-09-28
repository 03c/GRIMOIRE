#!/usr/bin/env python3
# k2_reference.py -- ground truth for K2-Horizon from its OWN modeling code
# (modeling_k2_horizon.py, trust_remote_code), on the CPU only: no GPU
# device is mapped into the container that runs this.  bf16 weights, up to
# MAXMEM in RAM, the rest read from the checkpoint on demand (accelerate).
#
# Prints the prompt's token ids (to compare with GRIMOIRE's), the top-5 next
# tokens after the prompt, and a short greedy continuation.
import os, sys, time, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = sys.argv[1] if len(sys.argv) > 1 else "/models/K2-Horizon-MoVA-36B-A4B"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "Explain in two sentences why the sky is blue."
NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MAXMEM = os.environ.get("MAXMEM", "44GiB")
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
msgs = [{"role": "user", "content": PROMPT}]
try:
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
except Exception as e:
    print("no chat template:", e)
    ids = tok(PROMPT, return_tensors="pt").input_ids
print("prompt ids:", ids.shape[1], ids[0].tolist(), flush=True)

t0 = time.time()
# Offload ONLY the routed experts and MoVA value experts to disk: they are
# always invoked through their own module forward, so accelerate's hooks
# load them.  Everything else stays in RAM -- K2's code reads
# v_router.bias directly (outside any forward), which an offloaded module
# leaves on the meta device ("Cannot copy out of meta tensor").
from accelerate import init_empty_weights
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(M, trust_remote_code=True)
with init_empty_weights():
    skel = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
# ONE entry per expert ModuleList, not one per expert leaf: transformers'
# expand_device_map scans every map entry for every parameter, so 16,380
# leaf entries kept the loader spinning for 15+ minutes before it read a
# single weight.
dm = {}
for name, mod in skel.named_modules():
    if list(mod.children()) or not list(mod.parameters(recurse=False)):
        continue
    for tag in (".mlp.experts.", ".v_experts."):
        if tag in name:
            dm[name[:name.index(tag) + len(tag) - 1]] = "disk"
            break
    else:
        dm[name] = "cpu"
del skel
print("device map:", sum(v == "cpu" for v in dm.values()), "cpu leaves,",
      sum(v == "disk" for v in dm.values()), "disk leaves", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    M, trust_remote_code=True, dtype=torch.bfloat16, device_map=dm,
    offload_folder="/tmp/k2-offload")
model.eval()
print(f"loaded in {time.time() - t0:.0f} s", flush=True)

with torch.no_grad():
    t0 = time.time()
    out = model(ids, use_cache=True)
    logits = out.logits[0, -1].float()
    top = torch.topk(logits, 5)
    print(f"prefill {time.time() - t0:.0f} s; top-5 next:",
          [(int(i), tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)],
          flush=True)
    gen = [int(top.indices[0])]
    past = out.past_key_values
    for _ in range(NEW - 1):
        o = model(torch.tensor([[gen[-1]]]), past_key_values=past, use_cache=True)
        past = o.past_key_values
        gen.append(int(o.logits[0, -1].argmax()))
    print("greedy ids:", gen)
    print("greedy text:", json.dumps(tok.decode(gen)), flush=True)
