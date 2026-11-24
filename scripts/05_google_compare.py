"""Origin–destination validation: our door-to-door estimate vs Google Maps (bus only) vs actual buses.

    python scripts/05_google_compare.py --make-template     # writes data/google_benchmark_template.csv
    python scripts/05_google_compare.py                     # compares data/google_benchmark.csv

CSV columns: origin_name,origin_lat,origin_lon,dest_name,dest_lat,dest_lon,weekday,depart_time,
             google_minutes,google_routes
For each row we plan the trip for the same weekday in the held-out week (25–31 Oct) and time, and
follow the real buses of our suggested routes through that day's GPS stop visits (±15 min of the
planned boarding, app/ridecheck.py), so each OD pair gets: ours vs Google vs what riding it really took.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

LANDMARKS = {  # approximate; verify by right-clicking in Google Maps
    "Kashmere Gate ISBT": (28.6675, 77.2282), "Nehru Place": (28.5494, 77.2519),
    "Anand Vihar ISBT": (28.6469, 77.3159), "Connaught Place": (28.6315, 77.2167),
    "Dwarka Sector 21": (28.5523, 77.0583), "Rohini Sector 18": (28.7380, 77.1390),
    "Mayur Vihar Phase 1": (28.6044, 77.2944), "Badarpur Border": (28.4934, 77.3030),
    "Narela": (28.8526, 77.0929), "Uttam Nagar": (28.6219, 77.0590), "AIIMS": (28.5672, 77.2100),
    "Lajpat Nagar": (28.5677, 77.2433), "Karol Bagh": (28.6519, 77.1909), "Azadpur": (28.7075, 77.1767),
    "Saket": (28.5244, 77.2066), "Janakpuri": (28.6219, 77.0878), "Shahdara": (28.6735, 77.2894),
    "ITO": (28.6287, 77.2410), "Mehrauli": (28.5196, 77.1849), "Okhla": (28.5355, 77.2732),
    "Punjabi Bagh": (28.6683, 77.1325), "Najafgarh": (28.6092, 76.9798),
    "New Delhi Railway Station": (28.6430, 77.2195), "Sarai Kale Khan ISBT": (28.5889, 77.2580),
    "Mundka": (28.6823, 77.0306),
}
PAIRS = [("Kashmere Gate ISBT", "Nehru Place"), ("Anand Vihar ISBT", "Connaught Place"),
         ("Dwarka Sector 21", "Connaught Place"), ("Rohini Sector 18", "Kashmere Gate ISBT"),
         ("Mayur Vihar Phase 1", "AIIMS"), ("Badarpur Border", "ITO"), ("Narela", "Azadpur"),
         ("Uttam Nagar", "Karol Bagh"), ("Lajpat Nagar", "Connaught Place"), ("Saket", "Nehru Place"),
         ("Janakpuri", "AIIMS"), ("Shahdara", "ITO"), ("Mehrauli", "AIIMS"), ("Okhla", "Lajpat Nagar"),
         ("Punjabi Bagh", "Connaught Place"), ("Najafgarh", "Dwarka Sector 21"),
         ("New Delhi Railway Station", "Nehru Place"), ("Sarai Kale Khan ISBT", "Anand Vihar ISBT"),
         ("Mundka", "Punjabi Bagh"), ("Azadpur", "Karol Bagh"), ("Nehru Place", "Kashmere Gate ISBT"),
         ("Connaught Place", "Anand Vihar ISBT"), ("Karol Bagh", "Saket"), ("ITO", "Janakpuri"),
         ("AIIMS", "Rohini Sector 18")]
TIMES = ["08:30", "13:00", "18:00", "22:00"]
WEEKDAY_DATE = {"sunday": 25, "monday": 26, "tuesday": 27, "wednesday": 28, "thursday": 29, "friday": 30,
                "saturday": 31}


def make_template():
    out = config.DATA / "google_benchmark_template.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["origin_name", "origin_lat", "origin_lon", "dest_name", "dest_lat", "dest_lon", "weekday",
                    "depart_time", "google_minutes", "google_routes"])
        for o, d in PAIRS:
            for tm in TIMES:
                w.writerow([o, *LANDMARKS[o], d, *LANDMARKS[d], "tuesday", tm, "", ""])
    print(f"template with {len(PAIRS) * len(TIMES)} rows → {out}\nFill google_minutes (door-to-door, bus only) "
          f"and save as {config.GOOGLE_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template", action="store_true")
    ap.add_argument("--csv", default=str(config.GOOGLE_CSV))
    a = ap.parse_args()
    if a.make_template:
        return make_template()
    from app import handlers as H
    from app.ridecheck import DayVisits
    st = H.state()
    pl = st["planner"]
    df = pd.read_csv(a.csv)
    visits: dict = {}
    rows = []
    for r in df.itertuples():
        day = WEEKDAY_DATE.get(str(r.weekday).strip().lower(), 27)
        dep = f"{config.YEAR}-{config.MONTH:02d}-{day:02d}T{str(r.depart_time).strip()[:5]}"
        rec = {"origin": r.origin_name, "dest": r.dest_name, "weekday": r.weekday, "time": r.depart_time,
               "google_min": float(r.google_minutes) if pd.notna(r.google_minutes) else np.nan}
        try:
            res = pl.plan(f"{r.origin_lat},{r.origin_lon}", f"{r.dest_lat},{r.dest_lon}", dep, "fastest", 3)
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)
            rows.append(rec)
            continue
        if not res["options"]:
            rec["error"] = "no route"
            rows.append(rec)
            continue
        o = res["options"][0]
        rec.update(ours_min=o["total_min"], ours_low=o["range"]["low_min"] if o.get("range") else np.nan,
                   ours_high=o["range"]["high_min"] if o.get("range") else np.nan,
                   timetable_min=o["timetable_total_min"], ours_routes=" + ".join(
                       l["route"] for l in o["legs"] if l["type"] == "bus"), transfers=o["transfers"])
        # reality on that held-out day for OUR legs: real buses of the same route that left the boarding
        # stop within ±15 min of the planned time (ride), and the real wait for the next one (app/ridecheck.py)
        if (config.SEG_DIR / f"visits_{config.day_name(day)}.parquet").exists():
            dv = visits.setdefault(day, DayVisits(day))
            pred_ride = act_ride = pred_wait = act_wait = 0.0
            n_obs, ride_ok, wait_ok = 0, True, True
            for leg in (l for l in o["legs"] if l["type"] == "bus"):
                P = pl.pat[leg["pattern"]]
                sids = list(P["stops"])
                b = sids.index(leg["board_stop"]["stop_id"])
                e = sids.index(leg["alight_stop"]["stop_id"], b + 1)
                seq = [int(c) for c in P["cl"][b:e + 1]]
                hms = leg["depart"][11:19]
                tb = int(hms[:2]) * 3600 + int(hms[3:5]) * 60 + int(hms[6:8])
                r_id = pl.pat_rint[leg["pattern"]]
                got = dv.rides(r_id, seq, tb, max_s=3 * leg["ride_s"] + 900)
                if not got:
                    ride_ok = False
                    break
                act_ride += float(np.median(got))
                n_obs += len(got)
                pred_ride += leg["ride_s"]
                w = dv.actual_wait(r_id, seq[0], tb - leg["wait_s"])
                if w is None:
                    wait_ok = False
                else:
                    act_wait += w
                    pred_wait += leg["wait_s"]
            if ride_ok:
                rec.update(ours_ride_min=round(pred_ride / 60, 1), actual_ride_min=round(act_ride / 60, 1),
                           actual_runs=n_obs)
                if wait_ok:
                    real = o["walk_min"] + (act_wait + act_ride) / 60
                    rec.update(real_door_to_door_min=round(real, 1))
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(config.MODEL_DIR / "google_comparison.csv", index=False)
    ok = out.dropna(subset=["ours_min", "google_min"]) if "ours_min" in out else out.iloc[0:0]
    lines = ["# Origin–destination comparison (auto-generated by scripts/05_google_compare.py)\n",
             f"{len(out)} queries, {len(ok)} with both our estimate and Google's.\n"]
    if len(ok):
        d = ok.ours_min - ok.google_min
        ape = (d.abs() / ok.google_min)
        lines += ["| metric | value |", "|---|---|",
                  f"| mean difference ours − Google | {d.mean():+.1f} min |",
                  f"| mean absolute difference | {d.abs().mean():.1f} min |",
                  f"| median abs. % difference | {100*ape.median():.1f} % |",
                  f"| within 20 % of Google | {100*(ape <= .2).mean():.0f} % |",
                  f"| correlation | {np.corrcoef(ok.ours_min, ok.google_min)[0,1]:.2f} |",
                  f"| Google inside our low–high range | {100*((ok.google_min >= ok.ours_low) & (ok.google_min <= ok.ours_high)).mean():.0f} % |"]
        bt = ok.groupby("time").apply(lambda x: pd.Series({"n": len(x), "ours": x.ours_min.mean(), "google": x.google_min.mean(),
                                                         "timetable": x.timetable_min.mean()})).round(1)
        lines += ["\n## By departure time (mean minutes)\n", "| time | n | ours | Google | timetable |", "|---|---|---|---|---|"]
        lines += [f"| {i} | {int(r.n)} | {r.ours} | {r.google} | {r.timetable} |" for i, r in bt.iterrows()]
    if "actual_ride_min" in out:
        ar = out.dropna(subset=["actual_ride_min"])
        if len(ar):
            e = ar.ours_ride_min - ar.actual_ride_min
            lines += ["\n## Our in-bus time vs real buses on the same legs (held-out day)\n",
                      f"{len(ar)} queries where real buses of our routes ran our legs that day "
                      f"({int(ar.actual_runs.sum())} bus runs): mean abs. error **{e.abs().mean():.1f} min**, "
                      f"median abs. % error **{100*(e.abs()/ar.actual_ride_min).median():.1f} %**."]
    if "real_door_to_door_min" in out:
        rd = out.dropna(subset=["real_door_to_door_min", "google_min", "ours_min"])
        if len(rd):
            eo = (rd.ours_min - rd.real_door_to_door_min).abs()
            eg = (rd.google_min - rd.real_door_to_door_min).abs()
            lines += ["\n## Door-to-door: who is closer to the real buses?\n",
                      f"{len(rd)} queries where every leg of our answer has real rides and well-detected real waits "
                      f"that day (walk + real wait + real ride). Google's own route may differ from ours, so this "
                      f"compares each estimate with what riding our suggestion really took.\n",
                      "| | mean abs. difference from real | closer to real |", "|---|---|---|",
                      f"| BusPulse | {eo.mean():.1f} min | {100 * (eo < eg).mean():.0f} % |",
                      f"| Google Maps | {eg.mean():.1f} min | {100 * (eg < eo).mean():.0f} % |"]
    lines += ["\n## All queries\n",
              "| from → to | time | ours | range | Google | timetable | real (our legs) | in-bus real vs ours | routes |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in out.itertuples():
        rng_ = f"{getattr(r, 'ours_low', '')}–{getattr(r, 'ours_high', '')}"
        act = f"{getattr(r, 'actual_ride_min', '')} vs {getattr(r, 'ours_ride_min', '')}" if pd.notna(getattr(r, "actual_ride_min", np.nan)) else ""
        real = getattr(r, "real_door_to_door_min", np.nan)
        lines.append(f"| {r.origin} → {r.dest} | {r.time} | {getattr(r, 'ours_min', getattr(r, 'error', ''))} | {rng_} | "
                     f"{r.google_min} | {getattr(r, 'timetable_min', '')} | {'' if pd.isna(real) else real} | {act} | "
                     f"{getattr(r, 'ours_routes', '')} |")
    (config.DOCS_DIR / "google_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:14]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(ok):
            fig, ax = plt.subplots(figsize=(5, 5))
            for tm, c in zip(TIMES, ["#c62828", "#0b6e4f", "#e39b00", "#3b6ea8"]):
                x = ok[ok.time.astype(str).str.startswith(tm[:2])]
                ax.scatter(x.google_min, x.ours_min, s=22, color=c, label=tm)
            lim = float(max(ok.google_min.max(), ok.ours_min.max())) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=1)
            ax.set_xlabel("Google Maps (bus) minutes"); ax.set_ylabel("BusPulse minutes")
            ax.legend(frameon=False, title="depart"); ax.set_title("Door-to-door: BusPulse vs Google")
            fig.tight_layout(); fig.savefig(config.DOCS_DIR / "img" / "google_scatter.png", dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
