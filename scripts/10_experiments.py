"""Accuracy experiments: train model variants on 1–24 Oct, score them on the SAME held-out chains (25–31 Oct).

Each variant writes its own model tables into data/tmp/exp/<name>/ (the real data/model is untouched),
is loaded with the production TravelTimeModel (same fallback chain), and is scored on the real bus
trip chains that scripts/04_validate.py uses. Writes docs/EXPERIMENTS.md.

    python scripts/10_experiments.py                 # all variants
    python scripts/10_experiments.py --only base recent14
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import gtfs as gtfs_mod  # noqa: E402
from app.calibration import build_chains  # noqa: E402
from app.traveltime import TravelTimeModel  # noqa: E402

EXP_DIR = config.TMP_DIR / "exp"
LAST_TRAIN = max(config.TRAIN_DAYS)

# name: (description, settings). Settings: days, tau (recency half-life in days, None = equal weights),
# lo/hi (winsorizing quantiles), outlier_k, stat (mean | median), shrink_k, scale (global multiplier)
VARIANTS = {
    "base": ("v1: all 24 days, winsorized mean 5–95 %, K=3", {}),
    "median": ("median instead of winsorized mean", {"stat": "median"}),
    "trim10": ("winsorize 10–90 %", {"lo": 0.10, "hi": 0.90}),
    "trim2": ("winsorize 2–98 %", {"lo": 0.02, "hi": 0.98}),
    "recent14": ("last 14 training days only", {"days": list(range(LAST_TRAIN - 13, LAST_TRAIN + 1))}),
    "recent7": ("last 7 training days only", {"days": list(range(LAST_TRAIN - 6, LAST_TRAIN + 1))}),
    "decay7": ("all days, recency weight half-life 7 days", {"tau": 7.0}),
    "decay14": ("all days, recency weight half-life 14 days", {"tau": 14.0}),
    "shrink1": ("less shrinkage to the all-hours value (K=1)", {"shrink_k": 1.0}),
    "shrink8": ("more shrinkage (K=8)", {"shrink_k": 8.0}),
    "trim15": ("winsorize 15–85 %", {"lo": 0.15, "hi": 0.85}),
    "trim20": ("winsorize 20–80 %", {"lo": 0.20, "hi": 0.80}),
    "trim25": ("winsorize 25–75 %", {"lo": 0.25, "hi": 0.75}),
    "trim15_hi90": ("winsorize 15–90 % (keep more of the slow tail)", {"lo": 0.15, "hi": 0.90}),
    "trim10_decay7": ("winsorize 10–90 % + recency half-life 7 days", {"lo": 0.10, "hi": 0.90, "tau": 7.0}),
    "trim10_decay14": ("winsorize 10–90 % + recency half-life 14 days", {"lo": 0.10, "hi": 0.90, "tau": 14.0}),
}


def train_tables(con, out: Path, s: dict, train_days=None):
    train_days = train_days or config.TRAIN_DAYS
    last = max(train_days)
    days = s.get("days", train_days)
    if "days" in s and train_days != config.TRAIN_DAYS:      # "last N days" variants, shifted to the split
        n = len(s["days"])
        days = list(range(last - n + 1, last + 1))
    tau = s.get("tau")
    lo, hi = s.get("lo", 0.05), s.get("hi", 0.95)
    k = s.get("outlier_k", 4.0)
    w = "1.0" if not tau else f"pow(0.5, ({last} - day) / {tau})"
    dl = ",".join(str(d) for d in days)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE t AS
        SELECT s.*, {w} AS w FROM s JOIN segmed USING (ca, cb)
        WHERE day IN ({dl}) AND run <= {k}*m + 60""")
    q = f"""
    WITH b AS (SELECT {{keys}}, quantile_cont(run, {lo}) AS r05, quantile_cont(run, {hi}) AS r95,
                      quantile_cont(dwell, {lo}) AS d05, quantile_cont(dwell, {hi}) AS d95
               FROM t GROUP BY {{keys}})
    SELECT {{keys}}, count(*) AS n, median(run) AS run_med,
           sum(w * least(greatest(run, r05), r95)) / sum(w) AS run_mean,
           median(dwell) AS dwell_med,
           sum(w * least(greatest(dwell, d05), d95)) FILTER (WHERE dwell IS NOT NULL)
             / sum(w) FILTER (WHERE dwell IS NOT NULL) AS dwell_mean,
           count(dwell) AS n_dwell
    FROM t JOIN b USING ({{keys}}) GROUP BY {{keys}}"""
    out.mkdir(parents=True, exist_ok=True)
    con.execute(q.format(keys="ca, cb, dt, h")).fetch_df().to_parquet(out / "seg_hour.parquet", index=False)
    con.execute(q.format(keys="ca, cb")).fetch_df().to_parquet(out / "seg_all.parquet", index=False)
    con.execute("""SELECT dt, h, count(*) AS n, median(run / m) AS factor,
                          median(p.dist / run * 3.6) AS speed_kmh, median(dwell) AS dwell_med
                   FROM t JOIN segmed USING (ca, cb) JOIN pairs p USING (ca, cb)
                   GROUP BY dt, h""").fetch_df().to_parquet(out / "city_hour.parquet", index=False)


class Chains:
    """The held-out chains of 04_validate.py (same sampling), flattened for fast scoring."""

    def __init__(self, tt, days, max_per_day=6000):
        self.items = []          # (pair indices, dt, t0, actual)
        self.live_seg = {}
        for day in days:
            f = config.SEG_DIR / f"seg_{config.day_name(day)}.parquet"
            if not f.exists():
                continue
            seg, wins = build_chains(pd.read_parquet(f), max_per_day)
            self.live_seg[day] = seg
            pis = np.array([tt.pair(a, b) for a, b in zip(seg.ca.to_numpy(), seg.cb.to_numpy())])
            dep_a, arr_b = seg.dep_a.to_numpy(), seg.arr_b.to_numpy()
            dt = int(seg.dt.iloc[0])
            for s, e in wins:
                p = pis[s:e]
                actual = float(arr_b[e - 1] - dep_a[s])
                if (p < 0).any() or actual <= 60:
                    continue
                self.items.append((p, dt, (dep_a[s] + config.IST_OFFSET_S) % 86400, actual, day))
        self.actual = np.array([x[3] for x in self.items])
        self.hour = np.array([int(x[2] // 3600) for x in self.items])

    def predict(self, tt, live=False):
        out = np.empty(len(self.items))
        feeds = {}
        for i, (p, dt, t0, _, day) in enumerate(self.items):
            if live:
                if day not in feeds:
                    from app.live import LiveFeed
                    feeds[day] = LiveFeed(tt, self.live_seg[day])
                tt.live = feeds[day]
                out[i] = tt.chain_time(p, dt, t0, now=t0)
            else:
                out[i] = tt.chain_time(p, dt, t0)
        tt.live = None
        return out


def score(actual, pred) -> dict:
    err = pred - actual
    ape = np.abs(err) / actual
    return {"MAE_min": np.abs(err).mean() / 60, "bias_min": err.mean() / 60, "MAPE_%": 100 * ape.mean(),
            "medAPE_%": 100 * np.median(ape), "within20_%": 100 * (ape <= 0.2).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--live", action="store_true", help="also score with the live blend (slower)")
    ap.add_argument("--inner", action="store_true",
                    help="model selection without touching the held-out week: train 1–17 Oct, score 18–24 Oct")
    ap.add_argument("--tune-live", metavar="VARIANT",
                    help="grid-search the live blend (K, decay, window) for one variant")
    a = ap.parse_args()
    train_days = list(range(1, 18)) if a.inner else config.TRAIN_DAYS
    test_days = list(range(18, 25)) if a.inner else config.TEST_DAYS
    g = gtfs_mod.load()
    base_tt = TravelTimeModel(g)
    t = time.time()
    ch = Chains(base_tt, test_days)
    print(f"{len(ch.items):,} chains on days {test_days} ({time.time() - t:.0f} s)")

    con = duckdb.connect()
    con.execute(f"SET temp_directory='{(config.TMP_DIR / 'duck').as_posix()}'")
    files = [config.SEG_DIR / f"seg_{config.day_name(d)}.parquet" for d in train_days]
    flist = "[" + ",".join(f"'{f.as_posix()}'" for f in files if f.exists()) + "]"
    con.register("pairs", pd.DataFrame({"ca": g["pairs"][:, 0], "cb": g["pairs"][:, 1], "dist": g["pair_dist"]}))
    con.execute(f"CREATE TABLE s AS SELECT ca, cb, dt, h, day, run, dwell FROM read_parquet({flist})")
    con.execute("CREATE TABLE segmed AS SELECT ca, cb, median(run) AS m FROM s GROUP BY ca, cb")
    print(f"training rows loaded ({time.time() - t:.0f} s)")

    if a.tune_live:
        tune_live(con, g, ch, a.tune_live, train_days)
        return
    names = a.only or list(VARIANTS)
    rows = []
    saved_k = config.SHRINK_K
    for name in names:
        desc, s = VARIANTS[name]
        t = time.time()
        out = EXP_DIR / name
        if out.exists():
            shutil.rmtree(out)
        train_tables(con, out, s, train_days)
        config.SHRINK_K = s.get("shrink_k", saved_k)
        tt = TravelTimeModel(g, model_dir=out, stat=s.get("stat", "mean"))
        config.SHRINK_K = saved_k
        pred = ch.predict(tt) * s.get("scale", 1.0)
        r = {"variant": name, "what": desc, **score(ch.actual, pred)}
        if a.live:
            r.update({f"live_{k}": v for k, v in score(ch.actual, ch.predict(tt, live=True)).items()
                      if k in ("MAPE_%", "within20_%")})
        rows.append(r)
        print(f"{name:10s} MAPE {r['MAPE_%']:.2f}  within20 {r['within20_%']:.1f}  MAE {r['MAE_min']:.2f}  "
              f"bias {r['bias_min']:+.2f}  ({time.time() - t:.0f} s)", flush=True)
    df = pd.DataFrame(rows).set_index("variant").round(2)
    tag = "inner" if a.inner else "heldout"
    df.to_csv(EXP_DIR / f"results_{tag}.csv")
    md = [f"## {'Model selection split' if a.inner else 'Held-out week'}: trained on {train_days[0]}–"
          f"{train_days[-1]} Oct, scored on {len(ch.items):,} real bus trip chains of {test_days[0]}–"
          f"{test_days[-1]} Oct\n",
          "| variant | what changed | MAE min | bias min | MAPE % | median APE % | within 20 % |"
          + (" live MAPE % | live within 20 % |" if a.live else ""),
          "|---|---|---|---|---|---|---|" + ("---|---|" if a.live else "")]
    for v, r in df.iterrows():
        md.append(f"| {v} | {r['what']} | {r['MAE_min']:.2f} | {r['bias_min']:+.2f} | {r['MAPE_%']:.2f} | "
                  f"{r['medAPE_%']:.2f} | {r['within20_%']:.1f} |"
                  + (f" {r['live_MAPE_%']:.2f} | {r['live_within20_%']:.1f} |" if a.live else ""))
    (EXP_DIR / f"section_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    write_doc()


HEADER = """# Accuracy experiments (auto-generated by scripts/10_experiments.py)

How the final settings were chosen without peeking at the held-out week:

1. **Model selection split** — variants are trained on 1–17 Oct and compared on real bus trips of 18–24 Oct.
   The live-blend weights are tuned on the same split.
2. **Held-out week** — the chosen setting (and, for transparency, every other variant) is then trained on
   1–24 Oct and scored once on 25–31 Oct. That week is never used to pick anything.

**Chosen (v2, in production):** `trim15` (winsorized mean 15–85 %: best MAPE / within-20 % on the selection
split without hurting MAE; heavier clipping starts to cost MAE) + live blend window 15 min, fade 60 min.
Recency weighting looked good on the held-out week but did not help on the selection split, so it was not used.

Lower MAPE / MAE and higher "within 20 %" are better. The bias is negative because bus trip times are
right-skewed (a few very slow trips); shifting predictions up removes the bias but makes MAPE and
"within 20 %" worse, so it is left as is.
"""


def write_doc():
    parts = [HEADER]
    for tag in ("inner", "live", "heldout"):
        f = EXP_DIR / f"section_{tag}.md"
        if f.exists():
            parts.append(f.read_text(encoding="utf-8"))
    (config.DOCS_DIR / "EXPERIMENTS.md").write_text("\n".join(parts), encoding="utf-8")


def tune_live(con, g, ch, variant, train_days):
    """Grid over live-blend K (pseudo-count), decay of the live weight with look-ahead, and window."""
    desc, s = VARIANTS[variant]
    out = EXP_DIR / f"live_{variant}"
    if out.exists():
        shutil.rmtree(out)
    train_tables(con, out, s, train_days)
    tt = TravelTimeModel(g, model_dir=out, stat=s.get("stat", "mean"))
    static = score(ch.actual, ch.predict(tt))
    saved = config.LIVE_K, config.LIVE_DECAY_S, config.LIVE_WINDOW_S
    from app.live import LiveFeed
    # one static pass: per chain element its base run, the dwell after it, and how far ahead of
    # "now" (= chain start) it is. The blend weights then apply vectorized (look-ahead from the
    # static times instead of the blended ones: a second-order difference).
    cid, base, dwell, ahead, pis, t0s, days = [], [], [], [], [], [], []
    for i, (p, dt, t0, _, day) in enumerate(ch.items):
        t = t0
        for k, pi in enumerate(p):
            r = tt.run_l[dt][int(t // 3600) % 24][pi]
            cid.append(i); base.append(r); ahead.append(t - t0); pis.append(pi); t0s.append(t0); days.append(day)
            t += r
            d = tt.dwell_l[dt][int(t // 3600) % 24][pi] if k < len(p) - 1 else 0.0
            dwell.append(d)
            t += d
    cid, base, dwell, ahead = map(np.asarray, (cid, base, dwell, ahead))
    pis, t0s, days = np.asarray(pis), np.asarray(t0s), np.asarray(days)
    rows = []
    for w_s in (900, 1800, 2700):
        config.LIVE_WINDOW_S = w_s
        med = np.full(len(cid), np.nan)
        n = np.zeros(len(cid))
        for day in np.unique(days):
            feed = LiveFeed(tt, ch.live_seg[day])
            idx = np.nonzero(days == day)[0]
            for j in idx:
                sn = feed.snapshot(t0s[j]).get(int(pis[j]))
                if sn is not None:
                    med[j], n[j] = sn
        has = n > 0
        for k in (1.0, 3.0, 8.0):
            for d in (900.0, 1800.0, 3600.0):
                w = np.where(has, n / (n + k) * np.exp(-ahead / d), 0.0)
                run = np.where(has, w * np.nan_to_num(med) + (1 - w) * base, base)
                pred = np.bincount(cid, weights=run + dwell, minlength=len(ch.items))
                r = score(ch.actual, pred)
                rows.append({"K": k, "decay_min": d / 60, "window_min": w_s / 60, **r})
                print(f"K={k:4.1f} decay={d/60:4.0f}m window={w_s/60:4.0f}m  MAPE {r['MAPE_%']:.2f}  "
                      f"within20 {r['within20_%']:.1f}", flush=True)
    config.LIVE_K, config.LIVE_DECAY_S, config.LIVE_WINDOW_S = saved
    df = pd.DataFrame(rows).sort_values("MAPE_%")
    best = df.iloc[0]
    md = [f"## Live blend tuning (model selection split, variant `{variant}`)\n",
          f"Static model without live data: MAPE {static['MAPE_%']:.2f} %, within 20 % {static['within20_%']:.1f}. "
          f"Current setting K={saved[0]:g}, decay {saved[1]/60:g} min, window {saved[2]/60:g} min. "
          f"Best: K={best.K:g}, decay {best.decay_min:g} min, window {best.window_min:g} min.\n",
          "| K | decay min | window min | MAPE % | within 20 % | MAE min |", "|---|---|---|---|---|---|"]
    for _, r in df.head(8).iterrows():
        md.append(f"| {r.K:g} | {r.decay_min:g} | {r.window_min:g} | {r['MAPE_%']:.2f} | "
                  f"{r['within20_%']:.1f} | {r.MAE_min:.2f} |")
    (EXP_DIR / "section_live.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    write_doc()


if __name__ == "__main__":
    main()
