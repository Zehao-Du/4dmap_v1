from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map4d.construction.SAM2 import (  # noqa: E402
    DEFAULT_SAM2_VERSION,
    SAM2Loader,
    default_sam2_checkpoints_root,
    sam2_validation_report_to_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SAM2 local path resolution/loading.")
    parser.add_argument("--version", default=DEFAULT_SAM2_VERSION, type=str)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--checkpoints-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--check-checkpoint-load",
        action="store_true",
        help="Run torch.load(map_location='cpu', weights_only=True) on the resolved checkpoint.",
    )
    parser.add_argument(
        "--check-model-instantiation",
        action="store_true",
        help="Build the base SAM2 model as a smoke test.",
    )
    parser.add_argument(
        "--instantiation-device",
        type=str,
        default=None,
        help="Optional device override for model instantiation; useful for forcing CPU validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loader = SAM2Loader(
        version=args.version,
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        checkpoints_root=args.checkpoints_root,
        device=args.device,
    )
    report = loader.validate(
        check_checkpoint_load=args.check_checkpoint_load,
        check_model_instantiation=args.check_model_instantiation,
        instantiation_device=args.instantiation_device,
    )
    print(sam2_validation_report_to_json(report))

    preferred_root = (
        pathlib.Path(args.checkpoints_root).expanduser().resolve()
        if args.checkpoints_root
        else default_sam2_checkpoints_root()
    )
    print(
        "\nExpected local convention:\n"
        f"  checkpoints: {preferred_root}\n"
        f"  checkpoint:  {report.checkpoint.checkpoint_path}\n"
        f"  config:      {report.checkpoint.config_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SAM2 validation failed: {exc}", file=sys.stderr)
        raise
