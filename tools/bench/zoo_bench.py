#!/usr/bin/env python3
"""tools/bench/zoo_bench.py GPU PORT MODEL[,MODEL...]  (BENCH_OUT=<dir> for results)

First used 2026-10-03 (HANDOFF-2026-10-03-SERVED.md).  -- served concurrency + prefix-cache sweep.

For every target checkpoint, against grimoire-server (bin/ on the host, run the
way the CLI baselines run -- tools/srv.sh hostbin mode):

  phase 1  server with GRIMOIRE_SEQ_SLOTS=8 (batching scheduler), prefix cache OFF
           - batch check: 4 prompts asked one at a time, then all 4 at once;
             greedy, so the texts must be identical
           - llama-benchy pp512/tg64 at concurrency 1 2 4 8
           - llama-benchy --enable-prefix-caching depth 4096 (the no-cache baseline)
           - prefix probe: two requests sharing a ~2K-token system context
  phase 2  same server + GRIMOIRE_PREFIX_CACHE=1, GRIMOIRE_PREFIX_SLOTS=8
           - the same prefix probe (second request should reuse the context,
             and must answer exactly as without the cache)
           - the same --enable-prefix-caching benchy

Writes zoo/<model>.json (everything measured) and prints one block per model.
"""
import csv, json, os, re, subprocess, sys, threading, time, urllib.request

B = os.environ.get('BENCH_OUT', '/mnt/storage/isos/grimoire-runs/bench')
Z = f'{B}/zoo'
SRV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'srv.sh')
BENCHY = ['uvx', 'llama-benchy@0.4.0']
os.makedirs(Z, exist_ok=True)

PROJ = {  # proj + extra server env, exactly as tools/regress_all_g0.sh runs them
    'Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE': ('mxfp4', []),
    'Ornith-1.5-35B-A3B': ('mxfp4', []),
    'Ornith-1.5-35B-A3B-FP8': ('mxfp4', []),
    'Ornith-1.5-35B-A3B-GPTQ-Int4': ('int4', []),
    'Ornith-1.5-35B-A3B-INT4-W4A16-AutoRound': ('int4', []),
    'Ornith-1.5-35B-A3B-MTPFIX': ('mxfp4', []),
    'Ornith-1.5-35B-A3B-NVFP4': ('mxfp4', []),
    'Qwen3.8-27B-MXFP4-GRIMOIRE': ('mxfp4', []),
    'Qwen3.8-27B-MXFP4-GRIMOIRE-MTPBF16': ('mxfp4', []),
    'Qwen3.8-27B': ('mxfp4', []),
    'Qwen3.8-27B-FP8': ('mxfp4', []),
    'Qwen3.8-27B-NVFP4': ('mxfp4', []),
    'Qwen3.8-27B-W4A16': ('int4', []),
    'Qwen3.8-27B-GPTQ-Int4-MTP-BF16': ('int4', []),
    'Qwen3.8-27B-MXFP4-AutoRound/Qwen3.8-27B-mxfp-w4g32': ('mxfp4', []),
    'Qwen3.8-27B-int4-AutoRound': ('int4', []),
    'Qwen3.6-35B-A3B-GPTQ-Int4': ('int4', []),
    'Muse-Glimmer-30B-INT4-W4A16': ('int4', []),
    'Muse-Glimmer-30B-GPTQ-INT4': ('int4', []),
    'Muse-Glimmer-30B-MXFP4': ('mxfp4', []),
    'Agnes-3.0-Flash': ('mxfp4', []),
    'K2-Horizon-MoVA-36B-A4B': ('mxfp4', []),
    'Qwen3.8-Flash-Next-NVFP4': ('bf16', ['GRIMOIRE_EXPERT_VRAM_PER_LAYER=112',
                                          'GRIMOIRE_PLE_FILE=/models/grimoire-ple/flash-next-ple.bin']),
}

PROMPTS = ["Explain in two sentences why the sky is blue.",
           "Write a haiku about a lighthouse.",
           "List three prime numbers greater than 100.",
           "What is the capital of Australia? Answer briefly."]
CTX = open('/mnt/storage/isos/grimoire-fuse/p4096.txt', errors='replace').read()[:12650]
Q1 = "Summarize the text above in one sentence."
Q2 = "Quote the first ten words of the text above."


def log(lane, msg):
    line = f'[{time.strftime("%H:%M:%S")}] {lane} {msg}'
    print(line, flush=True)


def run(cmd, env=None, timeout=None):
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return 124, f'TIMEOUT after {timeout}s\n{(e.stdout or b"")!s}'


def srv_up(name, gpu, port, model, proj, envlines):
    env = dict(os.environ, SRV_MODE='hostbin', SRV_ENV='\n'.join(envlines), SRV_WAIT='1200')
    return run(['bash', SRV, 'up', name, gpu, str(port), model, proj], env=env, timeout=1300)


def srv_down(name, port):
    return run(['bash', SRV, 'down', name, str(port)], timeout=400)


def chat(port, model, messages, max_tokens=48, timeout=900):
    body = json.dumps({"model": f"/models/{model}", "messages": messages,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f'http://localhost:{port}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {'error': f'HTTP {e.code}: {e.read()[:300]!r}', 's': time.time() - t0}
    except Exception as e:  # noqa
        return {'error': repr(e)[:300], 's': time.time() - t0}
    m = d['choices'][0].get('message', {})
    return {'text': (m.get('content') or ''), 'reasoning': m.get('reasoning_content') or '',
            'finish': d['choices'][0].get('finish_reason'), 'usage': d.get('usage', {}),
            's': time.time() - t0}


def key(r):
    return None if 'error' in r else (r['text'], r['reasoning'])


def batch_check(port, model):
    serial = [chat(port, model, [{"role": "user", "content": p}]) for p in PROMPTS]
    conc = [None] * len(PROMPTS)

    def go(i):
        conc[i] = chat(port, model, [{"role": "user", "content": PROMPTS[i]}])
    ts = [threading.Thread(target=go, args=(i,)) for i in range(len(PROMPTS))]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.time() - t0
    same = [key(a) is not None and key(a) == key(b) for a, b in zip(serial, conc)]
    return {'serial': serial, 'concurrent': conc, 'identical': same,
            'serial_wall_s': sum(r['s'] for r in serial), 'concurrent_wall_s': wall}


def complete(port, model, prompt, max_tokens=24, timeout=900):
    body = json.dumps({"model": f"/models/{model}", "prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f'http://localhost:{port}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {'error': f'HTTP {e.code}: {e.read()[:300]!r}', 's': time.time() - t0}
    except Exception as e:  # noqa
        return {'error': repr(e)[:300], 's': time.time() - t0}
    return {'text': d['choices'][0].get('text') or '', 'reasoning': '', 'usage': d.get('usage', {}),
            'finish': d['choices'][0].get('finish_reason'), 's': time.time() - t0}


def prefix_probe(port, model):
    """Three ways a client can repeat a long prefix; time the request that could reuse it.
    shared : same ~2K-token system context, a different user message (llama-benchy's pattern)
    chat   : turn 2 = turn 1 + the assistant's reply + a new question, through /v1/chat/completions
    raw    : the same continuation through /v1/completions (prompt + completion + more text)"""
    sys_msg = {"role": "system", "content": CTX}
    out = {}
    out['r1'] = chat(port, model, [sys_msg, {"role": "user", "content": Q1}], 32)
    out['r2'] = chat(port, model, [sys_msg, {"role": "user", "content": Q2}], 32)          # shared
    a1 = out['r1'].get('text', '')
    out['chat2'] = chat(port, model, [sys_msg, {"role": "user", "content": Q1},
                                      {"role": "assistant", "content": a1},
                                      {"role": "user", "content": Q2}], 32)                 # chat
    p1 = CTX + "\n\nQuestion: " + Q1 + "\nAnswer:"
    out['raw1'] = complete(port, model, p1, 24)
    p2 = p1 + out['raw1'].get('text', '') + "\n\nQuestion: " + Q2 + "\nAnswer:"
    out['raw2'] = complete(port, model, p2, 24)                                             # raw
    out['raw2_cold'] = complete(port, model, "Note.\n" + p2, 24)                           # same length, no reusable prefix
    return out


def tok_path(model):
    return f'/mnt/storage/Models/{model}'


def benchy(port, model, args, tag, timeout=5400):
    out = f'{Z}/{tag}.csv'
    if os.path.exists(out): os.remove(out)
    cmd = BENCHY + ['--base-url', f'http://localhost:{port}/v1', '--model', f'/models/{model}',
                    '--tokenizer', tok_path(model)] + args + ['--format', 'csv', '--save-result', out]
    rc, text = run(cmd, timeout=timeout)
    open(f'{Z}/{tag}.log', 'w').write(text)
    rows = list(csv.DictReader(open(out))) if os.path.exists(out) else []
    errs = sorted(set(re.findall(r'(HTTP \d{3}[^\n]{0,120}|Error[^\n]{0,160}|Coherence test FAILED)', text)))[:5]
    return {'rc': rc, 'rows': rows, 'errors': errs}


def fmt_rows(rows):
    out = []
    for r in rows:
        def f(k):
            v = r.get(k)
            return f'{float(v):9.1f}' if v not in (None, '', 'None') else '        -'
        out.append(f'      {r["test_name"]:<26} t/s {f("t_s_mean")}  req {f("t_s_req_mean")}  '
                   f'e2e_ttft(ms) {f("e2e_ttft_mean")}')
    return '\n'.join(out)


def sched_line(text):
    m = re.findall(r'scheduler:[^\n]*|prefix[^\n]*|unavailable[^\n]*|not usable[^\n]*', text)
    return ' | '.join(sorted(set(x.strip() for x in m)))[:400]


def one(gpu, port, model):
    proj, extra = PROJ[model]
    safe = model.replace('/', '__')
    res = {'model': model, 'proj': proj, 'gpu': gpu, 'start': time.strftime('%F %T')}
    name = f'z{port}'
    # ---------------- phase 1: batching, cache off ----------------
    slots = 8
    rc, text = srv_up(name, gpu, port, model, proj, extra + [f'GRIMOIRE_SEQ_SLOTS={slots}'])
    if rc != 0 and 'EXITED' in text:
        res['load8_error'] = text[-800:]
        slots = 1
        rc, text = srv_up(name, gpu, port, model, proj, extra)
    res['p1_ready'] = text.strip()[-600:]
    res['p1_slots'] = slots
    if rc != 0:
        res['fatal'] = 'server did not start'
        srv_down(name, port)
        return res
    guard('before ' + "res['batch']")
    res['batch'] = batch_check(port, model)
    guard('before ' + "res['conc']")
    # Muse's own batched path decodes at ~0.8 tok/s per row under the scheduler (measured
    # 2026-10-03: c1/c2/c4/c8 = 0.8/1.6/3.0/5.6, 43 min for one variant); c1+c2 is enough to
    # confirm the same behaviour on the other Muse checkpoints.
    levels = ['1', '2'] if 'Muse' in model else ['1', '2', '4', '8']
    res['conc_levels'] = levels
    res['conc'] = benchy(port, model, ['--pp', '512', '--tg', '64', '--concurrency', *levels,
                                        '--runs', '1' if 'Muse' in model else '2'], f'{safe}-conc')
    guard('before ' + "res['pfx_off_benchy']")
    res['pfx_off_benchy'] = benchy(port, model, ['--pp', '256', '--tg', '32', '--depth', '4096',
                                                 '--enable-prefix-caching', '--runs', '1' if 'Muse' in model else '2'], f'{safe}-pfxoff')
    guard('before ' + "res['pfx_off_probe']")
    res['pfx_off_probe'] = prefix_probe(port, model)
    srv_down(name, port)
    res['p1_log'] = sched_line(open(f'{B}/logs/{name}.log', errors='replace').read())
    os.replace(f'{B}/logs/{name}.log', f'{Z}/{safe}-p1.server.log')
    # ---------------- phase 2: prefix cache on ----------------
    rc, text = srv_up(name, gpu, port, model, proj,
                      extra + [f'GRIMOIRE_SEQ_SLOTS={slots}', 'GRIMOIRE_PREFIX_CACHE=1', 'GRIMOIRE_PREFIX_SLOTS=8'])
    res['p2_ready'] = text.strip()[-600:]
    if rc == 0:
        guard('before ' + "res['pfx_on_probe']")
        res['pfx_on_probe'] = prefix_probe(port, model)
        guard('before ' + "res['pfx_on_benchy']")
        res['pfx_on_benchy'] = benchy(port, model, ['--pp', '256', '--tg', '32', '--depth', '4096',
                                                    '--enable-prefix-caching', '--runs', '1' if 'Muse' in model else '2'], f'{safe}-pfxon')
    else:
        res['p2_error'] = text[-800:]
    srv_down(name, port)
    try:
        res['p2_log'] = sched_line(open(f'{B}/logs/{name}.log', errors='replace').read())
        os.replace(f'{B}/logs/{name}.log', f'{Z}/{safe}-p2.server.log')
    except OSError:
        pass
    res['end'] = time.strftime('%F %T')
    return res


def summary(res):
    m = res['model']
    lines = [f'== {m} ({res["proj"]}, {res["gpu"]})  slots={res.get("p1_slots")}']
    if 'fatal' in res:
        lines.append(f'   FATAL: {res["fatal"]}: {res.get("p1_ready","")[-300:]}')
        return '\n'.join(lines)
    lines.append(f'   server: {res.get("p1_log","")}')
    b = res.get('batch', {})
    errs = [r.get('error') for r in b.get('concurrent', []) + b.get('serial', []) if r and 'error' in r]
    lines.append(f'   batch check: identical {sum(b.get("identical", []))}/{len(PROMPTS)}  '
                 f'serial {b.get("serial_wall_s", 0):.1f}s vs 4-at-once {b.get("concurrent_wall_s", 0):.1f}s'
                 + (f'  ERRORS {errs[:2]}' if errs else ''))
    c = res.get('conc', {})
    lines.append(f'   concurrency (pp512/tg64) rc={c.get("rc")} {c.get("errors") or ""}')
    lines.append(fmt_rows(c.get('rows', [])))
    for tag in ('pfx_off_benchy', 'pfx_on_benchy'):
        p = res.get(tag)
        if p:
            lines.append(f'   {tag} (depth 4096, pp256/tg32) rc={p.get("rc")} {p.get("errors") or ""}')
            lines.append(fmt_rows(p.get('rows', [])))
    off, on = res.get('pfx_off_probe'), res.get('pfx_on_probe')
    if off and on:
        for k, label in (('r2', 'shared context'), ('chat2', 'chat turn 2'), ('raw2', 'raw continuation')):
            a, b = off.get(k, {}), on.get(k, {})
            same = key(a) is not None and key(a) == key(b)
            err = b.get('error') or a.get('error')
            lines.append(f'   prefix {label:<17}: {a.get("s", 0):5.2f}s off -> {b.get("s", 0):5.2f}s on'
                         f'  answer identical: {same}' + (f'  ERR {err[:120]}' if err else ''))
        if 'raw2_cold' in on:
            lines.append(f'   (raw cold control, cache on: {on["raw2_cold"].get("s", 0):.2f}s)')
    lines.append(f'   cache server: {res.get("p2_log","")}')
    return '\n'.join(lines)


def pci_of(gpu):
    r = subprocess.run(['bash', '/mnt/storage/isos/grimoire-fuse/tools/gpunode.sh', gpu],
                       capture_output=True, text=True)
    m = re.search(r'0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]', r.stderr)
    return m.group(0) if m else None


def aer_total(pci):
    """Sum of AER error counters on every port between the root complex and the card."""
    path = os.path.realpath(f'/sys/bus/pci/devices/{pci}')
    ports = [x for x in path.split('/') if re.fullmatch(r'0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]', x)]
    tot = 0
    for p in ports:
        for f in ('aer_dev_correctable', 'aer_dev_nonfatal', 'aer_dev_fatal'):
            try:
                for line in open(f'/sys/bus/pci/devices/{p}/{f}'):
                    if line.startswith('TOTAL_ERR'):
                        tot += int(line.split()[1])
            except OSError:
                pass
    return tot


class LinkTrouble(Exception):
    pass


GUARD = {}


def guard(where):
    """Stop the lane at the first new AER error or xe driver complaint on this card."""
    pci = GUARD.get('pci')
    if not pci:
        return
    a, bad = aer_total(pci), gpu_errors(pci)
    if a > GUARD['aer'] or len(bad) > GUARD['dmesg']:
        raise LinkTrouble(f'{where}: AER {GUARD["aer"]}->{a}, driver lines {GUARD["dmesg"]}->{len(bad)} '
                          f'{bad[GUARD["dmesg"]:][-2:]}')


def gpu_errors(pci):
    """xe/AER complaints about this card since boot (count + last lines)."""
    out = subprocess.run(['dmesg'], capture_output=True, text=True).stdout.splitlines()
    bad = [l for l in out if pci and pci[5:] in l and re.search(r'error|wedge|reset|fail|timed? ?out', l, re.I)
           and 'PCODE Mailbox' not in l]
    return bad


if __name__ == '__main__':
    gpu, port, models = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(',')
    pci = pci_of(gpu)
    GUARD.update(pci=pci, aer=aer_total(pci), dmesg=len(gpu_errors(pci)))
    log(gpu, f'lane on {pci}, {len(models)} models, AER {GUARD["aer"]}, {GUARD["dmesg"]} pre-existing driver lines')
    for m in models:
        try:
            guard('before ' + m)
        except LinkTrouble as e:
            log(gpu, f'STOP: {e}')
            break
        log(gpu, f'start {m}')
        try:
            r = one(gpu, port, m)
        except LinkTrouble as e:
            # do NOT stop the server: the card may be mid-submission or gone
            log(gpu, f'STOP (server z{port} left running, card needs a look): {e}')
            print(f'LANE {gpu} STOPPED', flush=True)
            sys.exit(3)
        except Exception as e:  # noqa -- keep the lane going, record why
            r = {'model': m, 'proj': PROJ.get(m, ('?',))[0], 'gpu': gpu, 'fatal': repr(e)[:500]}
            srv_down(f'z{port}', port)
        json.dump(r, open(f'{Z}/{m.replace("/", "__")}.json', 'w'), indent=1, default=str)
        print(summary(r), flush=True)
        log(gpu, f'done {m}')
    print(f'LANE {gpu} ALL DONE', flush=True)
