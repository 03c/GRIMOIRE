# vllm_mtp_fix.py -- used by vllm_mtp_up.sh (competitor container only).
# Resolve the one merge-conflict hunk left in my-vllm-xpu's installed qwen3_dflash.py
# (inside the throwaway container only): keep the v0.30.0 side.
p = '/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py'
s = open(p).read()
if '<<<<<<< HEAD\n' in s:
    a = s.index('<<<<<<< HEAD\n'); b = s.index('=======\n', a); c = s.index('>>>>>>> v0.30.0\n', b)
    s = s[:a] + s[b + len('=======\n'):c] + s[c + len('>>>>>>> v0.30.0\n'):]
    open(p, 'w').write(s)
    print('resolved conflict in', p)
