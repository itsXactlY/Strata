# The tape config (RTX 4060 Ti 16 GB, 32 GB RAM, Daedalus on top)

`strata-tape.json` + `run-strata-tape.sh` (both git-ignored, regenerate from this page). Engine: 0.1.39 merged with the
tape work (`engine/strata`; the previous binary is `engine/strata.pre-0.1.39-merge`).

```
--pack coder-iq1_m  --native <coder-IQ1_M gguf 1/2>  --ple-gguf <gguf 2/2>  --expert-profile data/expert-profile-coder.bin
--expert-cache auto  --prefill auto  --spec 4  --mtp Strata-data/mtp/rt  --spec-min-p 0.5
--max-context 65536  --kv int8  --kv-resident 32768  --vram-reserve-mib 1400
--conversation-cache-mib 4096  --conversation-cache-slots 2  --conversation-cache-min-free-mib 1024
--resident-budget-gib 20  --prompt-cache 12          (NOT --prompt-cache-every 8192, see below)
env: STRATA_KV_TAPE_HOST=1          (no STRATA_KV_TAPE_DEBUG, no STRATA_KV_EJECT_HOST_RELEASE, no "parallel")
```

| setting | why |
|---|---|
| `--max-context 65536` | the window Daedalus shows (65.5K); the rebase lands at ~32k |
| `--kv int8 --kv-resident 32768` | the live tape (rebase point) stays in VRAM; a bigger resident band starves the expert cache (56000 cost 35% of it); int8 avoids the k8v4 + streaming bug (#1188) |
| `--vram-reserve-mib 1400` | the Mazemaker embedding worker sits on the same GPU |
| `--conversation-cache-* 4096 / 2 / 1024` | two parked conversations; parking measured cheap (0.2-0.4 s per ~16k tokens), `parallel: 2` measured worse (below) |
| `--resident-budget-gib 20` | an upper bound: the engine grants what RAM allows (available - 4 GiB - margin; 19.1 GiB on a quiet box, pinned 15.55 GiB). This is the decode-speed lever |
| `--prompt-cache 12` | checkpoints are the ONLY resume points; more leaves survive longer. Keep `--prompt-cache-every` at its default (16384): see the warning below |

## Measured 2026-10-11 (`tools/tape_slot_bench.py`, ~30.6k-token tape, 58-token side call between turns)

| | warm turn after a side call | side call | rebase (same head, summary, last turns) | decode at 30k | swapped out / PSI avg60 |
|---|---|---|---|---|---|
| default checkpoints (6 / 16384), budget 20 | 2.0-2.2 s | 1.2-1.3 s | **63.5 s, 53% reused, 14,259 tokens re-read** | - | 2796 MiB / 15.7 |
| this config | 1.9-2.2 s | 1.3-1.6 s | **4.3-8.4 s, 100% reused, ~100 tokens re-read** | **31.8-47.0 tok/s** | 1 MiB (short run) / 6.4 |
| same, budget 12 | 2.2-2.5 s | 1.7-1.9 s | 8.4 s, 100% | 23.4-25.2 tok/s | 381 MiB / 2.8 |
| merged engine, `"parallel": 2` | 2.9-3.8 s | 1.5-7.2 s | - | - | 2 slots = 1.13 GiB VRAM, expert cache -12% |

Reuse was 100% on every warm turn. `kv_tape_parity` (37 tests / 241 assertions) and `kv_stream_verify` pass on the merged tree.

## Warning: do not set `--prompt-cache-every` below the system+tools length

The engine pins the OLDEST checkpoint for good. In Daedalus the fixed prefix (system 3.9k + tools 8.5k) is 12,638 tokens,
so with the default cadence (16384) the first checkpoint is the end of that prefix and every rebase resumes at 12,638.
With `--prompt-cache-every 8192` the first checkpoint is at 8,192 and is the one pinned: live rebases then reused only 8,192
and re-read 7-11k tokens in 37-48 s (strata-tape.log, 2026-10-11). The bench above missed this because its system prompt
was 30k tokens. The 4-8 s rebase in the table is that bench's, NOT a Daedalus rebase.

## Notes

- Memory is the thing to watch: with the engine loaded, MemAvailable sits near 4 GB. Lower `--resident-budget-gib` trades decode
  speed (12 GiB halves it) for headroom; the better lever is freeing RAM elsewhere.
- Checkpoints cost ~118 MB each of host RAM (12 = ~1.4 GB).
- Parked conversations: the live panel showed "parked 4 / restored 0" and parks of 1.4-12.7 s (not the 0.2-0.4 s of the quiet
  bench); a 33k-token snapshot is ~2.0 GB, so 2 slots reserve ~4 GB on a box under memory pressure. One slot is enough to
  park main while a side call runs; whether to keep any is open.
- The bench's rebase diverges at the end of a 30k system prompt; Daedalus's real rebase diverges after a ~12.6k system+tools
  head plus its kept head messages, so the gain there depends on a checkpoint surviving at that position. Not measured with
  a live Daedalus session yet.
- The 32-38 s park seen in the old live log was not reproducible with either engine; memory pressure is the suspect.

## 2026-10-14: the holygoat engine as the Daedalus tape (`strata-holygoat.json`)

Engine `engine/strata` = holygoat (806442f, no DROP_PARKED: Daedalus asks once and stops). Served by
`daedalus doctor start strata` (`MAIN_PROFILE=strata` in `~/.daedalus/stack.conf`).

```
holygoat args + --max-context 40960 --vram-reserve-mib 800
                + --conversation-cache-mib 3072 --conversation-cache-slots 1 --conversation-cache-min-free-mib 1024
                + --prompt-cache 8
"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}   (the GGUF's; without it every request was greedy)
```

| | value |
|---|---|
| expert arena | full resident (16.19 GiB RAM, 4443 GPU slots, no file reads). At 1400 MiB reserve / 65536 ctx it fell 1.92 GiB short (available RAM - fixed 4 GiB headroom) and decode dropped to 24-27 tok/s |
| decode at ~31k, varied code, 600-token answers | 34-44 tok/s (temp 1.0: 34-40, temp 0.7: 33-44, greedy: 37-40); drafts 63-70% accepted |
| the old 50-60 tok/s | repetitive bench text, ~96% of drafts accepted: not reachable on varied output on this box |
| side call with main parked | 0.6-1.0 s (park ~0.3 s, restore 75-85 ms), reuse 100% |
| rebase (bench) | 4.1 s, 99% reused |
| Daedalus | window 40960, compression.threshold 0.83 -> rebase at 34.0k, ~7k left for a reply |
