"""Prove a run makes no network calls.

Runs the real CLI in this process with a Python audit hook that records every
outgoing socket connect and DNS lookup. Any attempt to reach a non-loopback
address is listed and the script exits with status 1.

The model must already be on disk, because the first download is the one
moment arsub does use the network.

    python scripts/check_offline.py video.mp4 -m small -f srt
"""

from __future__ import annotations

import sys

attempts: list[str] = []
LOCAL = ("127.", "::1", "localhost")


def is_local(host: object) -> bool:
    return str(host).startswith(LOCAL)


def hook(event: str, args: tuple) -> None:
    if event == "socket.connect":
        address = args[1]
        host = address[0] if isinstance(address, tuple) and address else address
        if not is_local(host):
            attempts.append(f"connect {address}")
    elif event == "socket.getaddrinfo":
        if args[0] and not is_local(args[0]):
            attempts.append(f"dns lookup {args[0]}")


def main() -> int:
    sys.addaudithook(hook)
    from arsub.cli import main as cli

    code = cli(sys.argv[1:])
    if code != 0:
        print(f"arsub exited with {code}", file=sys.stderr)
        return code
    if attempts:
        print(f"FAILED: {len(attempts)} network attempts")
        for item in sorted(set(attempts)):
            print(f"  {item}")
        return 1
    print("OK: no network attempts during the run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
