"""Command-line entry point.

    python -m informality build    # download, estimate, model, check, report
    python -m informality export   # CSVs of the rates, model scores and checks
"""

from __future__ import annotations

import argparse
import csv
import sys

from .config import DOCS_DIR, EXPORT_DIR, REPORT_DIR, ensure_dirs
from .pipeline import run
from .report import render


def cmd_build(args: argparse.Namespace) -> int:
    ensure_dirs()
    print("loading EPHC microdata ...")
    analysis = run(reuse_cache=not args.refresh)
    head = analysis.headline

    print(f"\n{analysis.config['universe']['employment_category']['label']}")
    for label, n in analysis.universe.steps:
        print(f"  {n:8,d}  {label}")

    print(f"\n  informality rate      {head.value:.2%}  "
          f"({head.population:,.0f} workers)")
    print(f"  unweighted            {head.unweighted:.2%}  "
          f"<- the error an unweighted analysis would publish")
    print(f"  naive 95% interval    [{head.naive_low:.2%}, {head.naive_high:.2%}]")
    print(f"  design 95% interval   [{head.design_low:.2%}, {head.design_high:.2%}]"
          f"  ({head.design_effect:.2f}x wider)")
    print(f"  n={head.n:,}  clusters={head.clusters:,}  "
          f"effective n={head.effective_sample_size:,}")

    print("\n  models (5-fold, weighted)")
    for name, model in analysis.models.items():
        print(f"    {name:20s} AUC clustered {model.mean('clustered','auc'):.4f}  "
              f"random {model.mean('random','auc'):.4f}  "
              f"optimism {model.optimism:+.4f}")

    print("\n  feature contributions (AUC given up when removed)")
    for c in analysis.contributions:
        print(f"    {c.feature:14s} {c.loss:+.4f}")

    summary = analysis.summary
    print(f"\nchecks: {summary['total']} run, {summary['passed']} passed, "
          f"{summary['failed']} failed ({summary['blocking']} blocking)")
    for check in analysis.checks:
        if not check.passed:
            print(f"  [{check.severity:7}] {check.check_id}")
            print(f"            expected {check.expected}; observed {check.observed}")

    path = render(analysis, REPORT_DIR / "informality-report.html")
    print(f"\nreport: {path}")
    if args.publish_docs:
        published = DOCS_DIR / "index.html"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"published: {published}")

    if summary["blocking"]:
        print(f"\nFAILED: {summary['blocking']} blocking check(s)", file=sys.stderr)
    return 1 if summary["blocking"] else 0


def cmd_export(args: argparse.Namespace) -> int:
    ensure_dirs()
    analysis = run()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for column, group in analysis.by_group.items():
        for row in group:
            rows.append({"variable": column, **row})
    _write(EXPORT_DIR / "rates_by_group.csv",
           ["variable", "group", "n", "rate", "low", "high", "suppressed"], rows)

    _write(EXPORT_DIR / "model_scores.csv",
           ["model", "scheme", "fold", "auc", "brier", "n_test"],
           [{"model": name, "scheme": s.scheme, "fold": s.fold, "auc": s.auc,
             "brier": s.brier, "n_test": s.n_test}
            for name, m in analysis.models.items() for s in m.scores])

    _write(EXPORT_DIR / "feature_contributions.csv",
           ["feature", "auc_without", "auc_full", "loss"],
           [{"feature": c.feature, "auc_without": c.auc_without,
             "auc_full": c.auc_full, "loss": c.loss}
            for c in analysis.contributions])

    _write(EXPORT_DIR / "checks.csv",
           ["check_id", "family", "severity", "passed", "expected", "observed"],
           [{"check_id": c.check_id, "family": c.family, "severity": c.severity,
             "passed": c.passed, "expected": c.expected, "observed": c.observed}
            for c in analysis.checks])

    print(f"wrote 4 files to {EXPORT_DIR}")
    return 0


def _write(path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="informality")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="download, estimate, model and report")
    build.add_argument("--publish-docs", action="store_true",
                       help="also write docs/index.html for GitHub Pages")
    build.add_argument("--refresh", action="store_true",
                       help="re-download the microdata instead of using the cache")
    build.set_defaults(func=cmd_build)

    export = sub.add_parser("export", help="write CSVs")
    export.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
