"""python -m tokenbudget ..."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="tokenbudget", description="TokenBudget analysis console")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ui = sub.add_parser("ui", help="serve the local web console")
    p_ui.add_argument("--port", type=int, default=8000)
    p_ui.add_argument("--host", default="127.0.0.1")

    args = ap.parse_args()

    if args.cmd == "ui":
        import uvicorn

        uvicorn.run("tokenbudget.ui:create_app", factory=True, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
