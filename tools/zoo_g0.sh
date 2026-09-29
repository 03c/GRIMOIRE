#!/bin/bash
# zoo_g0.sh -- one short generation per checkpoint that the pure-mode
# regression (regress_all_g0.sh) does not cover yet.  gpu0 only, via
# g0run.sh (stops at the first new AER/xe error).  Prints speed and the
# start of the text so garbage is visible at a glance.
set -u
cd /mnt/storage/isos/grimoire-fuse
OUT=/mnt/storage/isos/grimoire-runs/zoo-$(date +%m%d-%H%M); mkdir -p "$OUT"; echo "results: $OUT"
P="Explain in two sentences why the sky is blue."
txt() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids'; }
while read -r m pr; do
  [ -z "$m" ] && continue
  r=$(LIM=900 bash tools/g0run.sh zoo-$m -m /models/$m --proj $pr --ctx 8192 -p "$P" -n 48)
  rc=$?
  cp /tmp/grim-zoo-$m.log "$OUT/$m.log" 2>/dev/null
  [ $rc -eq 9 ] && { echo "STOP: gpu0 guard tripped at $m"; exit 9; }
  printf "%-42s %-6s %s\n" "$m" "$pr" "${r:-NO RESULT LINE}"
  grep -hE "load: |generation failed|error|refus" "$OUT/$m.log" | grep -v unavailable | head -2 | sed 's/^/      /'
  echo "      text: $(txt $OUT/$m.log | tr '\n' ' ' | cut -c1-170)"
done <<'LIST'
Muse-Glimmer-30B-GPTQ-INT4 int4
Muse-Glimmer-30B-MXFP4 mxfp4
Ornith-1.5-35B-A3B-FP8 mxfp4
Ornith-1.5-35B-A3B-GPTQ-Int4 int4
Ornith-1.5-35B-A3B-INT4-W4A16-AutoRound int4
Qwen3.6-35B-A3B-GPTQ-Int4 int4
Qwen3.8-27B-GPTQ-Int4-MTP-BF16 int4
Qwen3.8-27B-int4-AutoRound int4
Qwen3.8-27B mxfp4
Ornith-1.5-35B-A3B mxfp4
Ornith-1.5-35B-A3B-MTPFIX mxfp4
Qwen3.8-27B-int4-ov int4
LIST
echo "ALL DONE"
