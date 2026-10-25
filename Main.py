"""Command-line entry point for the temperature reconstruction project.

Examples::

    python main.py              # run every stage (about 10 minutes)
    python main.py --quick      # fast smoke run (about 2 minutes)
    python main.py audit        # data audit and overview figures only
    python main.py tune validate reconstruct

Stages run in the order given; ``all`` (the default) runs them all.
Outputs go to ``results/`` (tables, figures, RESULTS.md).
"""
from __future__ import annotations

import argparse
import time

from tempfill.experiments import (Config, audit, load_params, performance, reconstruct,
                                  tune, validate, write_summary)

STAGES = ["audit", "tune", "validate", "performance", "reconstruct"]


def main(argv=None) -> None:
    """Parse arguments and run the requested stages."""
    parser = argparse.ArgumentParser(description="MMA3001 temperature reconstruction")
    parser.add_argument("stages", nargs="*", choices=STAGES + ["all"],
                        help="stages to run (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="fewer repeats, for a fast check that everything runs")
    args = parser.parse_args(argv)
    stages = STAGES if not args.stages or "all" in args.stages else args.stages
    cfg = Config(quick=args.quick)

    t_start = time.perf_counter()
    print("Stage: audit (always needed to load data)")
    ctx = audit(cfg)
    print(f"  primary {cfg.primary}: {ctx.stats[cfg.primary]['slots']} slots, "
          f"{ctx.stats[cfg.primary]['missing']} missing")

    params = load_params(cfg)
    results = {}
    if "tune" in stages:
        print("Stage: tune (development period only)")
        params = tune(cfg, ctx)
        print(f"  K={params['k_daily']}, J={params['k_weekly']}, window={params['window_days']} d, "
              f"switch={params['switch_len']} slots; final={params['final_method']}, "
              f"deployed={params['deployed']}")
    if "validate" in stages:
        print("Stage: validate (held-out test period + independent sensor)")
        results["val"] = validate(cfg, ctx, params)
    if "performance" in stages:
        print("Stage: performance")
        results["perf"] = performance(cfg, ctx, params)
    if "reconstruct" in stages:
        print("Stage: reconstruct real gaps")
        val = results.get("val")
        summaries = ({cfg.primary: val["primary"], cfg.secondary: val["secondary"]}
                     if val else None)
        results["recon"] = reconstruct(cfg, ctx, params, summaries)
    if all(k in results for k in ("val", "perf", "recon")):
        path = write_summary(cfg, ctx, params, results["val"], results["perf"], results["recon"])
        print(f"Summary written to {path}")
    print(f"Done in {time.perf_counter() - t_start:.0f} s. Outputs in {cfg.root / 'results'}")


if __name__ == "__main__":
    main()
