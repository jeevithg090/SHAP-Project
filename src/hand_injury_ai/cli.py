"""Command-line interface: ``hand-injury-ai {run,task-a,task-b,summary}``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from . import pipeline, summary, task_a, task_b
from .config import Paths
from .dataset import DataInputError

logger = logging.getLogger("hand_injury_ai")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_summary(paths: Paths, run_pipeline: bool) -> int:
    exit_code = pipeline.run_full_pipeline(paths) if run_pipeline else 0
    report = summary.build_summary(paths, pipeline_invoked=run_pipeline)
    json_path, md_path = summary.write_reports(report, paths.reports_dir)
    print(f"Summary JSON: {json_path}")
    print(f"Summary Markdown: {md_path}")
    print(f"Overall status: {report['overall_status']}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hand-injury-ai",
        description="SHAP feature-importance and AI composite scoring pipeline for hand injuries.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", parents=[common], help="Run Task A and Task B end to end.")
    sub.add_parser(
        "task-a", parents=[common], help="Build the pilot cohort, EDA and SHAP feature rankings."
    )
    sub.add_parser(
        "task-b", parents=[common], help="Train the AI composite model from Task A outputs."
    )
    summary_parser = sub.add_parser(
        "summary", parents=[common], help="Write the objective-level status report."
    )
    summary_parser.add_argument(
        "--run-pipeline", action="store_true", help="Run the full pipeline before summarising."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    paths = Paths.from_env()
    try:
        if args.command == "run":
            return pipeline.run_full_pipeline(paths)
        if args.command == "task-a":
            task_a.run(paths)
            return 0
        if args.command == "task-b":
            task_b.run(paths)
            return 0
        return _run_summary(paths, args.run_pipeline)
    except (DataInputError, task_b.TaskBInputError, task_a.QuickModelGateError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
