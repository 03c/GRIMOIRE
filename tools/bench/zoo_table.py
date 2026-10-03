#!/usr/bin/env python3
"""zoo_table.py [zoo_dir] -- one markdown table from zoo_bench.py's per-model JSON."""
import glob, json, os, re, sys
Z = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('BENCH_OUT', '/mnt/storage/isos/grimoire-runs/bench'), 'zoo')

def rows(res, key):
    return {r['test_name']: r for r in (res.get(key) or {}).get('rows', [])}

def num(r, k, scale=1.0, fmt='{:.1f}'):
    v = (r or {}).get(k)
    return fmt.format(float(v) * scale) if v not in (None, '', 'None') else '-'

out = ['| model | fmt | sched | c1 | c2 | c4 | c8 | c8/c1 | batch==serial | pfx chat off→on (s) | pfx raw off→on (s) | benchy pp256@d4096 TTFT off→on (ms) |',
       '|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|']
for f in sorted(glob.glob(os.path.join(Z, '*.json'))):
    r = json.load(open(f))
    m = r['model'].split('/')[0]
    if 'fatal' in r:
        out.append(f'| {m} | {r.get("proj")} | FATAL: {str(r["fatal"])[:60]} |' + ' |' * 9); continue
    c = rows(r, 'conc')
    tg = {k: c.get(f'tg64 (c{k})') for k in (1, 2, 4, 8)}
    t = {k: num(tg[k], 't_s_mean') for k in tg}
    try: ratio = f'{float(tg[8]["t_s_mean"]) / float(tg[1]["t_s_mean"]):.2f}x'
    except Exception: ratio = '-'
    sched = 'batch' if 'batching up to' in (r.get('p1_log') or '') else ('1-at-a-time' if 'one request' in (r.get('p1_log') or '') else '?')
    b = r.get('batch', {})
    ident = f'{sum(b.get("identical", []))}/{len(b.get("identical", []))}' if b else '-'
    off, on = r.get('pfx_off_probe') or {}, r.get('pfx_on_probe') or {}
    def pr(k):
        a, z = off.get(k, {}), on.get(k, {})
        if 'error' in a or 'error' in z: return 'ERR'
        return f'{a.get("s", 0):.2f}→{z.get("s", 0):.2f}' if a and z else '-'
    poff, pon = rows(r, 'pfx_off_benchy'), rows(r, 'pfx_on_benchy')
    k = next((x for x in poff if x.startswith('pp256 @ d')), None)
    bt = f'{num(poff.get(k), "e2e_ttft_mean", 1, "{:.0f}")}→{num(pon.get(k), "e2e_ttft_mean", 1, "{:.0f}")}' if k else '-'
    out.append(f'| {m} | {r.get("proj")} | {sched} | {t[1]} | {t[2]} | {t[4]} | {t[8]} | {ratio} | {ident} | {pr("chat2")} | {pr("raw2")} | {bt} |')
print('\n'.join(out))
