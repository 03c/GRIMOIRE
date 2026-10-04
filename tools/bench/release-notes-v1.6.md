**Long prompts under concurrency: 8 users × 4096-token prompts now decode at 390 tok/s total (v1.5: 74). Short-prompt concurrency and single-user speed are unchanged.**

Measured with llama-benchy 0.4.0 on one Arc Pro B70, Ornith-1.5-35B-A3B-MXFP4-GRIMOIRE, `GRIMOIRE_SEQ_SLOTS=8`. Totals are tok/s.

**4096-token prompts, 32 tokens generated** (pp4096/tg32):

| concurrent requests | v1.5 decode | **v1.6 decode** | v1.6 prompt processing |
|---|---:|---:|---:|
| 1 | 171.5 | **171.1** | 9,883 |
| 2 | 213.7 | **212.4** | 9,617 |
| 4 | 94.1 | **296.4** | 10,022 |
| 8 | 74.1 | **389.8** | 10,324 |

**512-token prompts, 64 tokens generated** (pp512/tg64), unchanged from v1.5:

| concurrent requests | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| decode | 193.4 | 237.9 | 355.2 | 466.2 |

Single request on the default server is unchanged: decode at tg256 is 195.5 tok/s at ~0 context, 183.5 at 2K and 172.3 at 4K. Prefill is ~9,900 tok/s at 4K.

### What changed ([bec7d2c](https://github.com/doopeworld/GRIMOIRE/commit/bec7d2c))
v1.5 admitted at most 8,192 prompt tokens in one batched prefill. Only two 4096-token prompts fit, so the other six were admitted one after another. The first requests then waited ~2.5 s after their first token before decoding resumed.

- **One prefill per burst.** Simultaneous prompts now go through one prefill of up to **32,768 tokens**. Eight 4096-token prompts take one 3.2-second prefill and then decode together.
- **Overflow batches.** Prompts beyond the cap go into further batches, not one at a time.
- **Setting.** Change the cap with `GRIMOIRE_ADMIT_BATCH_TOKENS`. The default is 32768; 0 admits one prompt at a time.

### Use
```
curl -LO https://github.com/doopeworld/GRIMOIRE/releases/download/v1.6/grimoire-b70-v1.6.tar.gz
docker load < grimoire-b70-v1.6.tar.gz          # -> grimoire-b70:latest
docker run -d --name grimoire --network host --init --device /dev/dri/renderD128 \
  -e GRIMOIRE_SEQ_SLOTS=8 \
  -v /path/to/models:/models grimoire-b70:latest \
  server --model /models/<checkpoint> --proj mxfp4 --ctx 8192 --port 8000
```
Find your B70's render node with `ls -l /dev/dri/by-path/`. The numbers can change after a reboot. Never set `SYCL_UR_USE_LEVEL_ZERO_V2=0`.

sha256 `699747d09d5fff9a431753b68b78de245162db275f5ed62fcbe561ef0d97928d`
