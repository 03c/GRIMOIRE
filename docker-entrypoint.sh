#!/bin/bash
# docker-entrypoint.sh -- dispatches to bin/grimoire-server or bin/grimoire,
# and ALWAYS wraps the real work in `timeout --signal=TERM --kill-after=60`
# so a `docker stop` gets a graceful TERM with time to drain GPU work
# before anything resorts to SIGKILL. A B70 that gets SIGKILLed mid-
# submission can fall off the PCI bus and need a power cycle -- this is
# not a hypothetical, it is this project's most repeated failure mode.
#
# That protects against a SLOW stop. It does NOT replace --init: without
# it, THIS script is PID 1, gets the kernel's default (do-nothing) signal
# disposition for most signals, and never reaps the binary's children --
# so `docker stop` still has no clean way to ask the workload to finish.
# Pass --init on `docker run` every time. The warning below fires if you
# forget.
set -u

if [ "$$" -eq 1 ]; then
    echo "WARNING: no --init on this container (docker-entrypoint.sh is PID 1)." >&2
    echo "         docker stop can only SIGKILL from here -- GPU work in flight" >&2
    echo "         may wedge the card. Re-run with 'docker run --init ...'." >&2
fi

TIMEOUT_SECS="${GRIMOIRE_TIMEOUT_SECS:-0}"   # 0 = no limit (a server should not have one)
run() {
    if [ "$TIMEOUT_SECS" -gt 0 ]; then
        exec timeout --signal=TERM --kill-after=60 "$TIMEOUT_SECS" "$@"
    else
        exec "$@"
    fi
}

case "${1:-server}" in
    server)
        shift
        run /grimoire/bin/grimoire-server "$@"
        ;;
    generate|cli)
        shift
        run /grimoire/bin/grimoire "$@"
        ;;
    /grimoire/bin/*|bin/*)
        run "$@"
        ;;
    *)
        echo "usage: docker run ... <image> {server|generate} [args...]" >&2
        echo "  server   -> bin/grimoire-server (OpenAI-compatible HTTP), e.g.:" >&2
        echo "              server --model /models/K2-Horizon-MoVA-36B-A4B --proj mxfp4 --ctx 8192 --port 8000" >&2
        echo "  generate -> bin/grimoire (one-shot CLI), e.g.:" >&2
        echo "              generate -m /models/K2-Horizon-MoVA-36B-A4B --proj mxfp4 --ctx 8192 -p \"...\" -n 256" >&2
        exit 2
        ;;
esac
