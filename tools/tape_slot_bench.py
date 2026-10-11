#!/usr/bin/env python3
"""Tape workload vs. a side call: what does a second conversation cost the first?

Daedalus keeps ONE long conversation append-only (the tape) and every so often sends a tiny unrelated request
(title generation, curator). On a one-conversation engine the side request parks the main conversation first
(~1.1 ms per cached token: 32-38 s at 30k tokens measured 2026-10-11) and the next main request restores it.
This drives exactly that against a running server and reports, per turn:

  main   : latency, prompt tokens, cached tokens (the reuse), prefill-visible time
  side   : latency of the tiny request
  main+1 : the main request right AFTER a side call (where parking/restore shows up)

  python3 tools/tape_slot_bench.py --url http://127.0.0.1:8091 --doc-repeat 150 --turns 4
Start the server twice, once without and once with "parallel": 2 in the config, and compare the two tables.
"""
import argparse
import json
import sys
import time
import urllib.request

DOC = ("The history of the printing press begins with movable type in East Asia and continues with Gutenberg's "
       "press in Mainz around 1450, which spread across Europe within decades, lowered the cost of books, changed "
       "how scholars, merchants and churches shared knowledge, and fed the Reformation and the scientific "
       "revolution. ")


def chat(url, messages, max_tokens, timeout=900):
    body = json.dumps({"model": "strata", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "stream": False}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    dt = time.monotonic() - t0
    u = data.get("usage") or {}
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    text = (data["choices"][0]["message"].get("content") or "").strip()
    return {"s": dt, "prompt": u.get("prompt_tokens", 0), "cached": cached,
            "out": u.get("completion_tokens", 0), "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--doc-repeat", type=int, default=150, help="~160 tokens per repeat")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--no-side", action="store_true", help="control run: no side requests")
    ap.add_argument("--json", default="", help="write the raw rows here")
    a = ap.parse_args()

    messages = [{"role": "system", "content": "You are a terse assistant.\n" + DOC * a.doc_repeat}]
    rows = []
    prev_was_side = False
    for k in range(a.turns + 1):
        messages.append({"role": "user", "content": f"Step {k}: say one short sentence about the text."})
        m = chat(a.url, messages, a.max_new)
        messages.append({"role": "assistant", "content": m["text"] or "ok"})
        rows.append(dict(m, kind="main" + ("+1" if prev_was_side else ""), turn=k))
        print(f"turn {k} main{'+1' if prev_was_side else '  '}: {m['s']:7.1f}s  prompt {m['prompt']:6d}  "
              f"cached {m['cached']:6d} ({100 * m['cached'] / max(m['prompt'], 1):3.0f}%)", flush=True)
        prev_was_side = False
        if a.no_side or k == a.turns:
            continue
        s = chat(a.url, [{"role": "user", "content": f"Title for note {k}: "}], 8)
        rows.append(dict(s, kind="side", turn=k))
        print(f"turn {k} side  : {s['s']:7.1f}s  prompt {s['prompt']:6d}  cached {s['cached']:6d}", flush=True)
        prev_was_side = True

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    base = [r["s"] for r in rows if r["kind"] == "main"][1:]          # the first turn is the cold read
    after = [r["s"] for r in rows if r["kind"] == "main+1"]
    side = [r["s"] for r in rows if r["kind"] == "side"]
    reuse = [r["cached"] / max(r["prompt"], 1) for r in rows if r["kind"].startswith("main")][1:]
    print(f"\nmain (warm, no side before): {mean(base):.1f}s   main right after a side call: {mean(after):.1f}s   "
          f"side call: {mean(side):.1f}s   mean reuse {100 * mean(reuse):.0f}%")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
