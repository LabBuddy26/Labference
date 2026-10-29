"""Run reference collection for the independent release protocols.

Each worker keeps the release controller and physical success oracle.  The
runner's ``--dataset-output`` writes the current LabBench HDF5 schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


TASKS = {
    "centrifuge_easy": ("eai4s_eval.centrifuge_v2.cli", "eai4s_eval/centrifuge_v2/configs/easy.json"),
    "centrifuge_hard": ("eai4s_eval.centrifuge_v2.cli", "eai4s_eval/centrifuge_v2/configs/hard.json"),
    "weighing_easy": ("eai4s_eval.weighing_v1.cli", "eai4s_eval/weighing_v1/configs/easy.json"),
    "weighing_hard": ("eai4s_eval.weighing_v1.cli", "eai4s_eval/weighing_v1/configs/hard.json"),
    "rack_easy": ("eai4s_eval.rack_recall_v4.cli", "eai4s_eval/rack_recall_v4/configs/easy.json"),
    "rack_same_cap": ("eai4s_eval.rack_recall_v4.cli", "eai4s_eval/rack_recall_v4/configs/hard_same_cap.json"),
    "rack_distinct_cap": ("eai4s_eval.rack_recall_v4.cli", "eai4s_eval/rack_recall_v4/configs/hard_distinct_cap.json"),
    "device_operation_hard": ("eai4s_eval.device_operation_v1.cli", "eai4s_eval/device_operation_v1/configs/hard.json"),
}


def command(task: str, output_root: Path, seed: int, max_steps: int | None,
            save_video: bool) -> list[str]:
    module, config = TASKS[task]
    output = output_root / task / f"run_{seed}"
    dataset = output_root / task / f"dataset_{seed}"
    args = [sys.executable, "-m", module, "run", "--config", config,
            "--output", str(output), "--seed", str(seed), "--headless",
            "--dataset-output", str(dataset)]
    if max_steps is not None:
        args += ["--max-steps", str(max_steps)]
    if save_video:
        args.append("--save-video")
    return args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), action="append")
    parser.add_argument("--output-root", default="outputs/release_collection")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    selected = args.task or list(TASKS)
    root = Path(args.output_root)
    for task in selected:
        cmd = command(task, root, args.seed, args.max_steps, args.save_video)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
