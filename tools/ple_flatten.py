#!/usr/bin/env python3
# ple_flatten.py SRC_SAFETENSORS DST_FILE
#
# Write Qwen3.8-Flash-Next's PLE n-gram table (128 row shards
# ...ngram_embedding.shard_{i}.weight, FP8 E4M3 [rows_i][160]) as ONE flat
# file, rows in global order, so row r is at byte r * 160.  GRIMOIRE reads
# 16 rows per token from it (GRIMOIRE_PLE_FILE).  Put the file on a
# dataset with a small recordsize: ZFS always reads whole records, and at
# the default 128K a 160-byte row costs 128 KB of I/O.
import json, os, re, struct, sys, time

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
base = 8 + n
shards = {}
for k, v in hdr.items():
    m = re.search(r"ngram_embedding\.shard_(\d+)\.weight$", k)
    if m:
        shards[int(m.group(1))] = v
idx = sorted(shards)
assert idx == list(range(len(idx))), "shard indices are not contiguous"
width = shards[0]["shape"][1]
rows = sum(shards[i]["shape"][0] for i in idx)
print(f"{len(idx)} shards, {rows} rows x {width} B -> {dst}", flush=True)
t0 = time.time()
done = 0
with open(src, "rb") as fi, open(dst + ".part", "wb") as fo:
    for i in idx:
        a, b = shards[i]["data_offsets"]
        fi.seek(base + a)
        left = b - a
        while left:
            chunk = fi.read(min(left, 64 << 20))
            fo.write(chunk)
            left -= len(chunk)
            done += len(chunk)
        if i % 16 == 15:
            print(f"  shard {i + 1}/{len(idx)}  {done / 1e9:.1f} GB  "
                  f"{done / 1e9 / (time.time() - t0):.2f} GB/s", flush=True)
    fo.flush()
    os.fsync(fo.fileno())
os.rename(dst + ".part", dst)
meta = {"rows": rows, "width": width, "dtype": shards[0]["dtype"], "source": src}
with open(dst + ".json", "w") as f:
    json.dump(meta, f)
print(f"done: {done / 1e9:.1f} GB in {time.time() - t0:.0f} s", flush=True)
