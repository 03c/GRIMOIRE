#!/bin/bash
# srv.sh up NAME GPU PORT MODEL PROJ  -- start grimoire-server, wait for /health
# srv.sh down NAME [PORT]             -- wait until idle, stop gracefully, keep log
# env: SRV_MODE=hostbin (default: grimoire-b70 runtime + /mnt/.../bin mounted ro,
#      exactly how the CLI baselines run) | image (SRV_IMAGE's own binary, e.g. grimoire-b70)
#      SRV_ENV=newline-separated K=V, SRV_CTX (8192), SRV_WAIT (secs, 900),
#      SRV_BIN=host bin/ dir to mount (default: the grimoire-fuse checkout's bin/)
set -u
B="${BENCH_OUT:-/mnt/storage/isos/grimoire-runs/bench}"; mkdir -p $B/logs
cmd=$1; shift
case $cmd in
up)
  NAME=$1 GPU=$2 PORT=$3 MODEL=$4 PROJ=$5
  cd /mnt/storage/isos/grimoire-fuse
  NODE=$(bash tools/gpunode.sh "$GPU") || { echo "cannot resolve $GPU"; exit 2; }
  for p in /proc/[0-9]*; do
    if ls -l $p/fd 2>/dev/null | grep -q "/dev/dri/$NODE"; then echo "REFUSING: $NODE held by pid $(basename $p) $(cat $p/comm)"; exit 3; fi
  done
  docker rm "$NAME" >/dev/null 2>&1
  ENVARGS=(-e ONEAPI_DEVICE_SELECTOR=level_zero:gpu)
  while IFS= read -r kv; do [ -n "$kv" ] && ENVARGS+=(-e "$kv"); done <<< "${SRV_ENV:-}"
  if [ "${SRV_MODE:-hostbin}" = image ]; then
    docker run -d --name "$NAME" --network host --init --stop-timeout 300 --device /dev/dri/$NODE \
      -v /mnt/storage/Models:/models -v /mnt/storage/isos/grimoire-cache:/cache "${ENVARGS[@]}" \
      "${SRV_IMAGE:-grimoire-b70:latest}" server --model /models/$MODEL --proj $PROJ \
      --ctx ${SRV_CTX:-8192} --host 0.0.0.0 --port $PORT >/dev/null || exit 4
  else
    docker run -d --name "$NAME" --network host -w /grimoire --init --stop-timeout 300 --device /dev/dri/$NODE \
      -v ${SRV_BIN:-/mnt/storage/isos/grimoire-fuse/bin}:/grimoire/bin:ro -v /mnt/storage/isos/grimoire-fuse/tools:/grimoire/tools:ro \
      --tmpfs /opt/grimoire/lib -v /mnt/storage/Models:/models "${ENVARGS[@]}" \
      --entrypoint /grimoire/bin/grimoire-server "${SRV_IMAGE:-grimoire-b70:latest}" \
      --model /models/$MODEL --proj $PROJ --ctx ${SRV_CTX:-8192} --host 0.0.0.0 --port $PORT >/dev/null || exit 4
  fi
  W=${SRV_WAIT:-900}
  for i in $(seq 1 $((W/3))); do
    sleep 3
    if curl -s -m 2 http://localhost:$PORT/health 2>/dev/null | grep -q ok; then
      echo "ready $NAME on $NODE :$PORT after $((i*3))s"
      docker logs "$NAME" 2>&1 | grep -E "scheduler:|unavailable|GiB|slots|prefix" | head -6 | sed 's/^/    /'
      exit 0
    fi
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "EXITED during load:"; docker logs "$NAME" 2>&1 | tail -15; docker rm "$NAME" >/dev/null 2>&1; exit 1; }
  done
  echo "not ready in ${W}s (left running, NOT killed -- may be mid-load)"; exit 1;;
down)
  NAME=$1 PORT=${2:-}
  # never stop with GPU work in flight: wait until no client is connected to the port
  if [ -n "$PORT" ]; then
    for i in $(seq 1 120); do
      n=$(ss -Htn state established "( sport = :$PORT )" 2>/dev/null | wc -l)
      [ "$n" -eq 0 ] && break; sleep 2
    done
    sleep 2
  fi
  docker logs "$NAME" > $B/logs/$NAME.log 2>&1
  docker stop --time 120 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1; echo "stopped $NAME";;
esac
