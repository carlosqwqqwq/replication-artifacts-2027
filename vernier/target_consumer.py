"""同一实验容器内的独立 Target queue consumer。

Direct producer 同步读取 K1/QEMU reference 并完成 EMI/MCMC，然后只写 durable
Target queue；本入口为每个 Target 保持一个独立 worker。consumer 不向 producer
返回 admission 信号，也不负责 campaign drain/finalize；Target 结果按自身进度落盘。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.target_queue_service import TargetQueueCoordinator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run independent Target queue consumers")
    parser.add_argument("--method-dir", required=True, type=Path)
    parser.add_argument(
        "--target", dest="targets", action="append",
        help="Target id; repeat once per simulator consumer",
    )
    parser.add_argument(
        "--config", type=Path,
        help="Frozen experiment config used to derive the method's Target set",
    )
    parser.add_argument("--method", help="Framework method id used with --config")
    parser.add_argument(
        "--feedback-target", default="all",
        help="Consume all declared method Targets or one Target",
    )
    parser.add_argument("--owner", default="standalone")
    args = parser.parse_args(argv)
    target_ids = tuple(dict.fromkeys(str(value) for value in (args.targets or ()) if str(value)))
    if not target_ids and args.config is not None:
        try:
            config = json.loads(args.config.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as error:
            parser.error(f"cannot read --config: {error}")
        methods = config.get("framework_methods", ())
        method = next(
            (
                item for item in methods
                if isinstance(item, dict) and item.get("id") == args.method
            ),
            None,
        ) if isinstance(methods, (list, tuple)) else None
        declared = method.get("feedback_targets") if isinstance(method, dict) else None
        if args.feedback_target == "all":
            target_ids = tuple(
                str(item.get("id")) for item in config.get("targets", ())
                if isinstance(item, dict) and item.get("id") in (declared or ())
            )
        else:
            if isinstance(declared, (list, tuple)) \
                    and args.feedback_target not in declared:
                parser.error(
                    f"feedback target is not declared for {args.method}: "
                    f"{args.feedback_target}"
                )
            target_ids = (str(args.feedback_target),)
    if not target_ids:
        parser.error("at least one --target or a valid --config/--method is required")

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)

    coordinator = TargetQueueCoordinator(
        args.method_dir, target_ids, owner=args.owner, stream_only=True,
    )
    coordinator.start()
    try:
        # Signal handlers wake this wait directly; no periodic health poll is
        # needed. The producer owns publication; this process owns only the
        # Target queues and stops immediately on an explicit container stop.
        while not stop.wait():
            pass
    finally:
        coordinator.stop(drain=False)
    # Per-job target gaps are durable result records and do not kill the
    # worker.  A worker-loop error is different: leave a non-zero process
    # status for the supervisor while the worker-status sidecar identifies the
    # affected Target.
    return 1 if coordinator.worker_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
