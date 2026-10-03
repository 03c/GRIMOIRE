#!/bin/bash
# cli_depth.sh -- the CLI's own decode speed at the depths benchy uses (gpu0).
# Per-token = (elapsed_b - elapsed_a) / (gen_b - gen_a): prefill and fixed costs cancel.
set -u
cd /mnt/storage/isos/grimoire-fuse
B="${BENCH_OUT:-/mnt/storage/isos/grimoire-runs/bench}"; mkdir -p $B/logs
HERE="$(cd "$(dirname "$0")" && pwd)"
O=/models/Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE
S="Write a detailed story about a lighthouse keeper."
P2="$(head -c 12650 p4096.txt)"; P4="$(cat p4096.txt)"
r() { # name prompt n [env]
  local x; x=$(EXTRA_ENV="${4:-}" LIM=600 bash tools/g0run.sh "$1" -m $O --proj mxfp4 --ctx 8192 -p "$2" -n $3) || { echo "STOP at $1: $x"; exit 9; }
  cp /tmp/grim-$1.log $B/logs/cli-$1.log; echo "$x"
}
per() { python3 - "$1" "$2" <<'PY'
import re,sys
def g(s):
    m=re.search(r'prompt=(\d+) generated=(\d+) finish=(\w+) elapsed=([\d.]+)s',s); return m and (int(m[1]),int(m[2]),m[3],float(m[4]))
a=g(sys.argv[1]); b=g(sys.argv[2])
if not a or not b or b[1]==a[1]: print("   n/a",a,b); sys.exit()
ms=(b[3]-a[3])/(b[1]-a[1])*1000
print(f"   prompt={a[0]} gen {a[1]}->{b[1]} ({b[2]}): {ms:.2f} ms/token = {1000/ms:.1f} tok/s")
PY
}
for env in "" "GRIMOIRE_DECODE_SPLITS=128"; do
  tag=${env:+-s128}
  echo "== short prompt ${env:-default}"; a=$(r cs32$tag "$S" 32 "$env"); b=$(r cs160$tag "$S" 160 "$env"); per "$a" "$b"
  echo "== ~2K prompt ${env:-default}";  a=$(r c2k32$tag "$P2" 32 "$env"); b=$(r c2k160$tag "$P2" 160 "$env"); per "$a" "$b"
  echo "== ~4K prompt ${env:-default}";  a=$(r c4k32$tag "$P4" 32 "$env"); b=$(r c4k160$tag "$P4" 160 "$env"); per "$a" "$b"
done
echo ALL DONE
