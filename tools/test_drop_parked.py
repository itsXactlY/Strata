#!/usr/bin/env python3
"""DROP_PARKED: the server's side of "the conversation the engine moved away from is gone for good".

  python3 tools/test_drop_parked.py
"""
import io
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.server import StrataEngine  # noqa: E402


class _Stdin(io.StringIO):
    def flush(self):
        pass


class _Proc:
    def __init__(self):
        self.stdin = _Stdin()


def _engine(reply_lines):
    e = StrataEngine.__new__(StrataEngine)
    e.proc = _Proc()
    e.lines = queue.Queue()
    for line in reply_lines:
        e.lines.put(line)
    return e


def main():
    # the command goes out, and the engine's answer is parsed
    e = _engine(["OK DROP_PARKED 1 2038"])
    res = e.drop_parked()
    assert e.proc.stdin.getvalue() == "DROP_PARKED\n", e.proc.stdin.getvalue()
    assert res == {"ok": True, "dropped": 1, "released_mib": 2038}, res

    # unrelated lines in front are skipped, not mistaken for the answer
    e = _engine(["", "T 1234", "OK DROP_PARKED 0 0"])
    assert e.drop_parked() == {"ok": True, "dropped": 0, "released_mib": 0}

    # a malformed answer still reports success of the command, with nothing counted
    e = _engine(["OK DROP_PARKED"])
    assert e.drop_parked() == {"ok": True, "dropped": 0, "released_mib": 0}

    # no engine process: None, the handler turns it into a 503
    e = _engine([])
    e.proc = None
    assert e.drop_parked() is None
    print("test_drop_parked: ok")


if __name__ == "__main__":
    main()
