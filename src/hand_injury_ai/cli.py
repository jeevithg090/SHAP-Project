"""Command-line interface: ``hand-injury-ai {run,task-a,task-b,summary,check,demo}``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__, pipeline, summary, task_a, task_b
from .config import (
    ENV_DATA_DICTIONARY,
    ENV_FEATURE_LIST,
    ENV_OUTPUT_DIR,
    ENV_PATIENT_ID_SALT,
    ENV_PROJECT_ROOT,
    ENV_REPORTS_DIR,
    ENV_WORKBOOK,
    Paths,
)
from .dataset import DataInputError

logger = logging.getLogger("hand_injury_ai")

MIN_DEMO_PATIENTS = 40

DESCRIPTION = """\
Explainable ML pipeline for traumatic hand injuries.

Task A ranks the clinical findings that drive four severity scores (MHISS,
Atthakomol, MUES, Strickland) with Random Forests and SHAP. Task B combines the
scores into a 0-100 AI composite, compares regressors and checks the result
against IPS."""

EPILOG = f"""\
examples:
  hand-injury-ai demo                  try it on a synthetic cohort (no real data needed)
  hand-injury-ai check                 show resolved paths and whether inputs exist
  hand-injury-ai run                   Task A + Task B on your own workbook
  hand-injury-ai run --workbook data/registry.xlsx --dictionary data/dictionary.xlsx
  hand-injury-ai summary --run-pipeline

configuration (flags override environment variables, which override .env):
  {ENV_WORKBOOK:<34}--workbook
  {ENV_DATA_DICTIONARY:<34}--dictionary
  {ENV_OUTPUT_DIR:<34}--output-dir
  {ENV_REPORTS_DIR:<34}--reports-dir
  {ENV_FEATURE_LIST:<34}optional curated Task B feature list
  {ENV_PATIENT_ID_SALT:<34}optional secret for keyed pseudonyms
  {ENV_PROJECT_ROOT:<34}base for relative paths (default: cwd)

exit status: 0 success, 1 missing/invalid input or failed run, 2 usage error,
130 interrupted."""


# --------------------------------------------------------------------------- helpers
def _configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    fmt = (
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
        if verbose
        else "%(asctime)s %(levelname)-7s %(message)s"
    )
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", force=True)


def _project_root() -> Path:
    return Path(os.environ.get(ENV_PROJECT_ROOT) or Path.cwd()).resolve()


def _apply_overrides(paths: Paths, args: argparse.Namespace) -> Paths:
    root = _project_root()
    overrides: dict[str, Path] = {}
    for attr in ("workbook", "data_dictionary", "output_dir", "reports_dir"):
        value: Path | None = getattr(args, attr, None)
        if value is not None:
            value = value.expanduser()
            overrides[attr] = value if value.is_absolute() else root / value
    return dataclasses.replace(paths, **overrides) if overrides else paths


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _print_run_report(paths: Paths, elapsed: float) -> None:
    """Print a short, human-readable recap after a successful run."""
    task_a_summary = _read_json(paths.task_a_summary)
    task_b_summary = _read_json(paths.task_b_summary)
    counts = task_a_summary.get("row_counts") or {}
    lines = ["", f"Done in {elapsed:.0f}s."]
    if isinstance(counts, dict) and counts:
        lines.append(
            "  Cohort       {total} rows -> {pilot} pilot, {holdout} holdout "
            "({bilateral} bilateral)".format(
                total=counts.get("total_rows", "?"),
                pilot=counts.get("pilot_rows", "?"),
                holdout=counts.get("holdout_rows", "?"),
                bilateral=counts.get("bilateral_rows", "?"),
            )
        )
    targets = task_a_summary.get("task_a_targets_completed")
    if isinstance(targets, list):
        names = ", ".join(str(target) for target in targets)
        lines.append(f"  Task A       SHAP rankings for {len(targets)} scores: {names}")
    if task_b_summary:
        lines.append(
            "  Task B       best model {model}: test R² {r2:.3f}, MAE {mae:.2f}, "
            "out-of-fold R² {oof:.3f}".format(
                model=task_b_summary.get("best_model", "?"),
                r2=float(task_b_summary.get("best_model_test_r2", float("nan"))),
                mae=float(task_b_summary.get("best_model_test_mae", float("nan"))),
                oof=float(task_b_summary.get("best_model_oof_r2", float("nan"))),
            )
        )
    lines += [
        f"  Outputs      {_display(paths.output_dir)}/",
        f"  Reports      {_display(paths.reports_dir)}/",
        f"  Start with   {_display(paths.task_b_dir / 'B7_task_b_summary.md')}",
    ]
    print("\n".join(lines))


# --------------------------------------------------------------------------- commands
def _cmd_run(paths: Paths, _args: argparse.Namespace) -> int:
    started = time.perf_counter()
    exit_code = pipeline.run_full_pipeline(paths)
    if exit_code == 0:
        _print_run_report(paths, time.perf_counter() - started)
    else:
        print(
            f"\nRun failed; details in {_display(paths.pipeline_summary)}. "
            "Run `hand-injury-ai check` to verify your configuration.",
            file=sys.stderr,
        )
    return exit_code


def _cmd_task_a(paths: Paths, _args: argparse.Namespace) -> int:
    started = time.perf_counter()
    task_a.run(paths)
    print(
        f"\nTask A done in {time.perf_counter() - started:.0f}s. "
        f"Rankings: {_display(paths.task_a_dir)}/  Next: hand-injury-ai task-b"
    )
    return 0


def _cmd_task_b(paths: Paths, _args: argparse.Namespace) -> int:
    started = time.perf_counter()
    result = task_b.run(paths)
    print(
        f"\nTask B done in {time.perf_counter() - started:.0f}s. "
        f"Best model {result['best_model']} (test R² {result['best_model_test_r2']:.3f}). "
        f"Outputs: {_display(paths.task_b_dir)}/"
    )
    return 0


def _cmd_summary(paths: Paths, args: argparse.Namespace) -> int:
    exit_code = pipeline.run_full_pipeline(paths) if args.run_pipeline else 0
    report = summary.build_summary(paths, pipeline_invoked=args.run_pipeline)
    json_path, md_path = summary.write_reports(report, paths.reports_dir)
    print(f"Summary JSON:     {_display(json_path)}")
    print(f"Summary Markdown: {_display(md_path)}")
    print(f"Overall status:   {report['overall_status']}")
    return exit_code


def _cmd_check(paths: Paths, _args: argparse.Namespace) -> int:
    def row(label: str, status: str, detail: str) -> str:
        return f"  {label:<17}{status:<9}{detail}"

    ok = True
    lines = [f"hand-injury-ai {__version__} configuration", f"  Project root     {_project_root()}"]
    for label, path, env_var, flag in (
        ("Workbook", paths.workbook, ENV_WORKBOOK, "--workbook"),
        ("Data dictionary", paths.data_dictionary, ENV_DATA_DICTIONARY, "--dictionary"),
    ):
        if path.is_file():
            lines.append(row(label, "OK", _display(path)))
        else:
            ok = False
            lines.append(row(label, "MISSING", _display(path)))
            lines.append(row("", "", f"-> put the file there, set {env_var} or pass {flag}"))
    for label, path in (("Output dir", paths.output_dir), ("Reports dir", paths.reports_dir)):
        status = "exists" if path.is_dir() else "new"
        lines.append(row(label, status, _display(path)))
    if paths.feature_list_override is None:
        lines.append(row("Feature list", "-", "Task A shortlist (default)"))
    elif paths.feature_list_override.is_file():
        lines.append(row("Feature list", "OK", _display(paths.feature_list_override)))
    else:
        ok = False
        lines.append(row("Feature list", "MISSING", _display(paths.feature_list_override)))
    task_a_ready = paths.pilot_dataset.is_file() and paths.provisional_feature_list.is_file()
    lines.append(
        row(
            "Task A outputs",
            "present" if task_a_ready else "none",
            "task-b can run" if task_a_ready else "run task-a (or run) first",
        )
    )
    lines.append(
        row(
            "Patient IDs",
            "-",
            "keyed HMAC-SHA256 pseudonyms (salt set)"
            if paths.patient_id_salt
            else f"sequential P0001 codes ({ENV_PATIENT_ID_SALT} not set)",
        )
    )
    lines.append("")
    lines.append(
        "Ready: `hand-injury-ai run`"
        if ok
        else "Not ready: fix the MISSING items above (or try `hand-injury-ai demo`)."
    )
    print("\n".join(lines))
    return 0 if ok else 1


def _cmd_demo(paths: Paths, args: argparse.Namespace) -> int:
    from .synthetic import write_synthetic_inputs  # imported lazily: only needed here

    demo_dir: Path = args.dir.expanduser()
    if not demo_dir.is_absolute():
        demo_dir = _project_root() / demo_dir
    workbook, dictionary = write_synthetic_inputs(
        demo_dir / "data", n_patients=args.patients, seed=args.seed
    )
    print(
        f"Wrote a synthetic cohort of {args.patients} fictional patients (seed {args.seed}):\n"
        f"  {_display(workbook)}\n  {_display(dictionary)}"
    )
    demo_paths = dataclasses.replace(
        paths,
        workbook=workbook,
        data_dictionary=dictionary,
        output_dir=demo_dir / "outputs",
        reports_dir=demo_dir / "reports",
        feature_list_override=None,
    )
    if args.no_run:
        print(
            "\nRun the pipeline on it with:\n"
            f"  hand-injury-ai run --workbook {_display(workbook)} "
            f"--dictionary {_display(dictionary)} --output-dir {_display(demo_dir / 'outputs')} "
            f"--reports-dir {_display(demo_dir / 'reports')}"
        )
        return 0
    print()
    return _cmd_run(demo_paths, args)


COMMANDS = {
    "run": _cmd_run,
    "task-a": _cmd_task_a,
    "task-b": _cmd_task_b,
    "summary": _cmd_summary,
    "check": _cmd_check,
    "demo": _cmd_demo,
}


# --------------------------------------------------------------------------- parser
def _positive_patients(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from exc
    if number < MIN_DEMO_PATIENTS:
        raise argparse.ArgumentTypeError(
            f"need at least {MIN_DEMO_PATIENTS} patients for every model to have enough rows"
        )
    return number


def build_parser() -> argparse.ArgumentParser:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        prog="hand-injury-ai",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=formatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    output = argparse.ArgumentParser(add_help=False)
    group = output.add_argument_group("output")
    verbosity = group.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="store_true", help="show debug logging with module names"
    )
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="only show warnings, errors and the final recap"
    )

    inputs = argparse.ArgumentParser(add_help=False)
    group = inputs.add_argument_group("paths (override environment variables and .env)")
    group.add_argument(
        "--workbook", type=Path, metavar="PATH", help=f"source workbook [{ENV_WORKBOOK}]"
    )
    group.add_argument(
        "--dictionary",
        dest="data_dictionary",
        type=Path,
        metavar="PATH",
        help=f"data dictionary workbook [{ENV_DATA_DICTIONARY}]",
    )
    outputs_parent = argparse.ArgumentParser(add_help=False)
    group = outputs_parent.add_argument_group("output locations")
    group.add_argument(
        "--output-dir", type=Path, metavar="DIR", help=f"generated artefacts [{ENV_OUTPUT_DIR}]"
    )
    group.add_argument(
        "--reports-dir", type=Path, metavar="DIR", help=f"run summaries [{ENV_REPORTS_DIR}]"
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name: str, summary_text: str, parents: list[argparse.ArgumentParser], desc: str):
        return sub.add_parser(
            name,
            parents=[output, *parents],
            help=summary_text,
            description=desc,
            formatter_class=formatter,
        )

    add(
        "run",
        "run Task A and Task B end to end",
        [inputs, outputs_parent],
        "Build the pilot cohort, EDA and SHAP rankings (Task A), then train and evaluate\n"
        "the AI composite model (Task B). Always writes single_file_run_summary.json.",
    )
    add(
        "task-a",
        "build the cohort, EDA and SHAP feature rankings",
        [inputs, outputs_parent],
        "Task A only: cohort building, EDA, quick MHISS gate, Random Forest + SHAP per score\n"
        "and the cross-score feature shortlist.",
    )
    add(
        "task-b",
        "train the AI composite model from Task A outputs",
        [outputs_parent],
        "Task B only: reads the pilot dataset and feature list written by task-a from\n"
        "--output-dir and trains/evaluates the AI composite model.",
    )
    summary_parser = add(
        "summary",
        "write the objective-level status report",
        [inputs, outputs_parent],
        "Summarise which objectives have complete outputs, with AI-vs-IPS agreement metrics.",
    )
    summary_parser.add_argument(
        "--run-pipeline", action="store_true", help="run the full pipeline before summarising"
    )
    add(
        "check",
        "show resolved configuration and whether inputs exist",
        [inputs, outputs_parent],
        "Print where every input and output resolves to, without running anything.\n"
        "Exits with status 1 if a required input is missing.",
    )
    demo_parser = add(
        "demo",
        "generate a synthetic cohort and run the pipeline on it",
        [],
        "Write a fully fictional workbook and data dictionary (same layout as the real\n"
        "registry, random values) to DIR/data and run the full pipeline into\n"
        "DIR/outputs and DIR/reports. Safe to share: contains no real patient data.",
    )
    demo_parser.add_argument(
        "--dir", type=Path, default=Path("demo"), metavar="DIR", help="demo folder (default: demo)"
    )
    demo_parser.add_argument(
        "--patients",
        type=_positive_patients,
        default=90,
        metavar="N",
        help=f"number of fictional patients, at least {MIN_DEMO_PATIENTS} (default: 90)",
    )
    demo_parser.add_argument(
        "--seed", type=int, default=7, help="random seed for the synthetic data (default: 7)"
    )
    demo_parser.add_argument(
        "--no-run", action="store_true", help="only write the synthetic inputs"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose, args.quiet)
    paths = _apply_overrides(Paths.from_env(), args)
    try:
        return COMMANDS[args.command](paths, args)
    except (DataInputError, task_b.TaskBInputError, task_a.QuickModelGateError) as exc:
        logger.error("%s", exc)
        print("Run `hand-injury-ai check` to see the resolved configuration.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
