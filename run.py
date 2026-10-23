from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Two-stage Sobol outbreak early warning")
    parser.add_argument("command", choices=["prepare", "train", "test", "all", "status"])
    parser.add_argument("--run-id", required=True, help="Run identifier; use a new ID for a new experiment")
    parser.add_argument("--data-dir", type=Path, help="Daily city-year CSV directory; default: data/raw")
    parser.add_argument("--threads", type=int, default=6, help="Stage 1 threads per fit")
    parser.add_argument("--rebuild-inputs", action="store_true", help="Rebuild development inputs during prepare")
    args = parser.parse_args(argv)
    if sys.flags.optimize:
        parser.error("Do not use python -O; assertion checks must stay enabled")
    if args.rebuild_inputs and args.command != "prepare":
        parser.error("--rebuild-inputs is supported only by prepare")
    from common import paths
    paths.configure(args.run_id, data_dir=args.data_dir, threads=args.threads)
    if args.command == "status":
        target = paths.RUN_DIR / "run_status.json"
        if target.exists():
            print(json.dumps(json.loads(target.read_text(encoding="utf-8-sig")), ensure_ascii=False, indent=2))
        else:
            print("No status file for this run: " + str(paths.RUN_DIR))
        return 0
    from main import runner
    if args.rebuild_inputs:
        runner.assert_pre_test()
        from common.data_builder import build_development
        build_development(force=True)
    runner.execute(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
