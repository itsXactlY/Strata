# The tape config (RTX 4060 Ti 16 GB, 32 GB RAM, Daedalus on top)

`strata-tape.json` + `run-strata-tape.sh` (both git-ignored, regenerate from this page). Engine: 0.1.39 merged with the
tape work (`engine/strata`; the previous binary is `engine/strata.pre-0.1.39-merge`).

```
--pack coder-iq1_m  --native <coder-IQ1_M gguf 1/2>  --ple-gguf <gguf 2/2>  --expert-profile data/expert-profile-coder.bin
--expert-cache auto  --prefill auto  --spec 4  --mtp Strata-data/mtp/rt  --spec-min-p 0.5
--max-context 65536  --kv int8  --kv-resident 32768  --vram-reserve-mib 1400
--conversation-cache-mib 4096  --conversation-cache-slots 2  --conversation-cache-min-free-mib 1024
--resident-budget-gib 20  --prompt-cache 12  --prompt-cache-every 8192
env: STRATA_KV_TAPE_HOST=1          (no STRATA_KV_TAPE_DEBUG, no STRATA_KV_EJECT_HOST_RELEASE, no "parallel")
```

| setting | why |
|---|---|
| `--max-context 65536` | the window Daedalus shows (65.5K); the rebase lands at ~32k |
| `--kv int8 --kv-resident 32768` | the live tape (rebase point) stays in VRAM; a bigger resident band starves the expert cache (56000 cost 35% of it); int8 avoids the k8v4 + streaming bug (#1188) |
| `--vram-reserve-mib 1400` | the Mazemaker embedding worker sits on the same GPU |
| `--conversation-cache-* 4096 / 2 / 1024` | two parked conversations; parking measured cheap (0.2-0.4 s per ~16k tokens), `parallel: 2` measured worse (below) |
| `--resident-budget-gib 20` | an upper bound: the engine grants what RAM allows (available - 4 GiB - margin; 19.1 GiB on a quiet box, pinned 15.55 GiB). This is the decode-speed lever |
| `--prompt-cache 12 --prompt-cache-every 8192` | checkpoints are the ONLY resume points. With the default 6 and every 16384, the end of a kept head is rotated out within a few turns and every rebase falls back to the oldest checkpoint (the end of the system prompt) |

## Measured 2026-10-11 (`tools/tape_slot_bench.py`, ~30.6k-token tape, 58-token side call between turns)

| | warm turn after a side call | side call | rebase (same head, summary, last turns) | decode at 30k | swapped out / PSI avg60 |
|---|---|---|---|---|---|
| default checkpoints (6 / 16384), budget 20 | 2.0-2.2 s | 1.2-1.3 s | **63.5 s, 53% reused, 14,259 tokens re-read** | - | 2796 MiB / 15.7 |
| this config | 1.9-2.2 s | 1.3-1.6 s | **4.3-8.4 s, 100% reused, ~100 tokens re-read** | **31.8-47.0 tok/s** | 1 MiB (short run) / 6.4 |
| same, budget 12 | 2.2-2.5 s | 1.7-1.9 s | 8.4 s, 100% | 23.4-25.2 tok/s | 381 MiB / 2.8 |
| merged engine, `"parallel": 2` | 2.9-3.8 s | 1.5-7.2 s | - | - | 2 slots = 1.13 GiB VRAM, expert cache -12% |

Reuse was 100% on every warm turn. `kv_tape_parity` (37 tests / 241 assertions) and `kv_stream_verify` pass on the merged tree.

## Notes

- Memory is the thing to watch: with the engine loaded, MemAvailable sits near 4 GB. Lower `--resident-budget-gib` trades decode
  speed (12 GiB halves it) for headroom; the better lever is freeing RAM elsewhere.
- Checkpoints cost ~118 MB each of host RAM (12 = ~1.4 GB).
- The bench's rebase diverges at the end of a 30k system prompt; Daedalus's real rebase diverges after a ~12.6k system+tools
  head plus its kept head messages, so the gain there depends on a checkpoint surviving at that position. Not measured with
  a live Daedalus session yet.
- The 32-38 s park seen in the old live log was not reproducible with either engine; memory pressure is the suspect.
