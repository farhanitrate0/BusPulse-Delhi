# BusPulse Delhi — technical note

**Problem 1: real-time bus traffic model and journey planner** (with Problem 2's crowding as a layer).
Delhi's buses (DTC and cluster/DIMTS — the GPS feed sees ~2,200 of the 2,574 GTFS routes on a typical day) are
used as moving traffic sensors: 31 days of GPS (124 GB, 913 M fixes) become a model of how long a bus
takes on every stretch between two stops, for every hour of a weekday, Saturday and Sunday. A journey planner
uses that model instead of the timetable, blends in live observations, says how reliable each answer is and how
full the bus usually is.

## 1. Data

| Source | Content | Use |
|---|---|---|
| Bus GPS, 1–31 Oct 2026 | 31 SQLite DBs (124 GB unpacked): `vehicle_id, lat, lng, route_id, timestamp…` | travel times, waits, live feed (replay) |
| Static GTFS (DTC/DoT, v32) | 6,862 stops, 2,574 routes, 80,357 trips, stop_times, shapes | network, stop order, timetable baseline |
| ONDC e-tickets, Oct 2026 | 2.56 M tickets: route code, start/end stop **names**, time | crowding (Problem 2 layer) |

Train/test split: the model learns from **1–24 Oct**; **25–31 Oct (a full Sunday–Saturday week) is held out**
for validation and is also the "live" feed in the demo. 2 Oct (Gandhi Jayanti) and 20 Oct (Dussehra) are treated
as Sundays; a per-day speed profile printed by `02_build_traveltime.py` checks this.

The archive is processed **one day at a time**: extract one DB → clean → reduce to stop-to-stop observations
(a few MB) → delete the DB. Peak disk use stays near 25 GB and a laptop handles the whole month.

## 2. Linking GPS points to the network (map-matching to stops, not to roads)

We match buses to **stops** rather than to OpenStreetMap road links. Every stop is a fixed point on the bus
network, the planner works stop-to-stop, and GTFS already says which stops follow each other — so a pair of
consecutive stop visits *is* a road stretch, without needing a road graph.

1. **Stop clusters.** GTFS stops within 50 m are merged (greedy leader clustering, busiest stop first). This
   pools the two sides of a road and nearby bays: 6,862 stops → 4,457 clusters. Direction comes from the order
   of the two clusters, so both directions stay separate.
2. **Cleaning** (counts in `DATA_QUALITY.md`): parse text lat/lng/timestamps, drop rows outside the Delhi
   bounding box, de-duplicate on (vehicle, timestamp), remove **GPS jumps** (a fix whose speed in *and* out
   exceeds 100 km/h — the classic off-road spike), sort by vehicle and time.
3. **Densify.** Where two consecutive fixes are 80 m–1.5 km apart and ≤180 s apart, points are interpolated
   every 40 m, so a bus moving fast between sparse fixes is not missed at a stop it passed.
4. **Snap.** A KD-tree gives each point its nearest stop cluster within 50 m.
5. **Visits.** Consecutive points at the same cluster (gaps < 120 s) form one visit:
   arrival = first point, departure = last point.
6. **Stretch observations.** Between two consecutive visits of the same bus:
   `run = arrival(B) − departure(A)` and `dwell = departure(B) − arrival(B)`.
   An observation is kept only if A→B are **adjacent on some GTFS route pattern** (otherwise a stop was missed,
   or the bus detoured or crossed a road), there is **no data gap > 180 s inside** it, and its straight-line
   speed is 3–70 km/h and the run is ≤ 30 min. Dwell over 180 s is a layover and is not used as dwell.
7. **Stop visits** are saved too (`visits_*.parquet`, 98 M over the month): every pass of every bus at every stop
   cluster, with its route. They give real departures (for waits) and let us follow a bus through a whole ride
   (for validation and ride calibration, §3).
8. **Stop detection.** A bus is only "seen" at a stop if its GPS comes within 50 m of the cluster. At some stops
   (stop placed off the road, two clusters close together) most passes are missed. For every (route, stop) we
   compare the buses seen there with the route's best-detected stops; GPS waits are only used where it is ≥ 70 %.

## 3. Travel time on each stretch

**Waiting at stops and signals.** Time at a stop (dwell, inside the 50 m circle) is kept **separate** from the
running time between stops; signal delay is part of running time, since it depends on the stretch and hour, not
on the stop. Separating them means a passenger who gets off is not charged the dwell at their stop, and the
model sees which stretches are slow because of traffic and which stops are slow because of boarding.

**Aggregation** (DuckDB over the per-day files): for every (stretch, day type, hour) — n, median, winsorised
mean (15–85 %), p25/p75/p85 of run and dwell. Observations above 4× the stretch's own median are dropped first
(a bus out of service). A (winsorised) mean is used for prediction because means add up along a trip.
The winsorising level, recency weighting, shrinkage and median-vs-mean were chosen on a model-selection split
(train 1–17 Oct, score on real trips of 18–24 Oct) so the held-out week was never used to pick settings —
`docs/EXPERIMENTS.md` has every variant.

**Sparse roads and hours — the fallback chain** (every estimate carries its level, shown in the app as the
share of the trip "measured from real buses"):

| Level | Used when | Estimate |
|---|---|---|
| 3 — stretch + hour | ≥ 5 observations at that day type and hour | shrunk mean: `w·obs + (1−w)·level 2`, `w = n/(n+3)` |
| 2 — stretch, other hours | stretch observed at other hours | stretch median × city slowdown factor for that day type and hour |
| 1 — city speed | stretch never observed (rare route, or next to a stop the GPS seldom detects) | distance ÷ city median bus speed at that day type and hour |
| 0 — timetable | no GPS model at all | GTFS scheduled time |

So with little or no data the model degrades smoothly to "this stretch, typical shape of the day", then to
"Delhi at this hour", never to a fixed timetable speed.

**Waiting for the bus.** Expected wait from the **GPS**, not the timetable: from every observed departure of the
same route from the same stop (one per bus pass) we get the headways H and use `E[H²] / (2·E[H])` — the expected
wait of a passenger who turns up at random. It equals H/2 for perfectly regular buses and grows when buses
**bunch** (two together, then a long gap), which the app reports ("buses often come bunched"). Where the GPS has
no reliable wait for a stop, half the timetable headway is used (rare routes: time to the next scheduled bus).
The GPS shows that **real buses come less often than scheduled**: GPS headways are ~1.2× the timetable's
mid-route, and the expected wait is ~2× half the scheduled headway.

**Calibrating the planner on whole rides (training days only).** The planner's legs are checked against real
buses on 18–24 Oct (`app/ridecheck.py`): for a planned leg (route r, boarding stop A at time t, alighting stop B),
the real buses of route r that left A within ±15 min of t are followed through their stop visits to B.
* **Ride.** Real rides take longer than the sum of their clean stretches (long halts at timing points, jams
  slower than 3 km/h, stops the GPS missed): the median ratio actual/planned is 1.02 in the morning peak and 1.19
  in the evening peak. The planner multiplies each ride by the median ratio for its time of day; the ratio
  quantiles give the range.
* **Wait.** A stop the GPS detects poorly (or a terminal, where buses wait out of range) takes the GPS wait of
  the nearest well-detected stop of the same route pattern (≤ 3 stops away) — the same buses pass both. Real
  waits ÷ predicted: ×1.16 for GPS-based waits (the planner tends to pick the route whose wait looks shortest),
  and **×5.8 where only the timetable is known** — routes the GPS barely sees mostly run far less than scheduled.
  The planner applies both factors while routing, so it steers riders towards routes that actually run.

**Reliability range.** Ride × the whole-ride ratio quantiles (relative to their median) for that time of day, plus
0.5–1.6× the expected wait — "usually 37–41 min; on 9 of 10 days you arrive by 09:13". On the held-out week the
ride ranges cover 43.9 % (claimed 50 %), 79.9 % (85 %), 89.9 % (90 %) — `RIDE_VALIDATION.md`.

## 4. Live + historical

`app/live.py` keeps recent stretch observations. For a stretch the bus will reach at time `t` when it is now
`now`:

```
live = median run of observations on this stretch in the last 15 min      (n of them)
w    = n / (n + 3) · exp(−(t − now) / 60 min)
est  = w · live + (1 − w) · historical(day type, hour)
```

(Window, K and fade were grid-searched on the model-selection split; results are flat around the optimum.)

Live data dominates the next few stretches and fades to history further ahead; no live data → pure history.
The demo replays the held-out Tuesday 27 Oct as the live feed (`live=true`). What it adds, measured honestly:
on the next 5–30 stops (stretch chains) MAPE falls from 15.9 % to 14.4 %; on whole rides (median 22 min) it is
roughly neutral (16.6 % → 16.5 %), because most of a long ride lies beyond what the last 15 minutes can tell.

**Real GTFS-RT feed** (`app/gtfsrt.py`, switched on by `BUSPULSE_GTFSRT_URL`): every poll (default 15 s) the
connector decodes the VehiclePositions feed with a built-in protobuf decoder (cross-checked against Google's
`gtfs-realtime-bindings`). It keeps the last 20 minutes of fixes per bus and runs them through **the same
code as the historical pipeline** (`app/gpsproc.py`: jumps, gaps, densify, stop visits, stop-to-stop runs).
Late fixes are sorted into place, repeated fixes are ignored, and a gap of more than 3 minutes never produces
a run. Network errors back off and never reach the API; `/health` reports polls, buses and runs. Checked
without a key by replaying 27 Oct's raw GPS as a GTFS-RT feed (`scripts/16_gtfsrt_check.py`,
[LIVE_CONNECTOR.md](LIVE_CONNECTOR.md)). The connector sees 4,200–4,500 stretches in its 15-minute window, and
its medians match the offline pipeline's to within 0–1 % (correlation 0.998). **On the real feed**
(Transport Stack `realtime_gtfs` endpoint, `x-api-key` header, 24 Nov 2026) it tracked 5,424 buses and
measured 649 stop-to-stop runs in its first three polls, with fixes typically 37 s old.

**Little or no live data**: a stretch with no live run in the last 15 minutes uses pure history; with n
live runs, history keeps a weight of 3 / (n + 3). Further ahead in the trip, the live weight fades within
the hour. If the feed is down, the planner gives exactly the historical answer.

## 5. Journey planner

* **Frequency-based, time-dependent RAPTOR** over stop clusters, up to 2 transfers. In-vehicle time is
  evaluated stretch by stretch at the clock time the bus reaches it, so a trip starting at 08:30 that runs into
  09:15 uses 09:00 speeds for its later stretches.
* Walking: ≤ 800 m to/from stops (1.25 m/s, ×1.3 detour; widened to 1.5 km with a warning), ≤ 300 m between
  stops when changing, 60 s to cross the road at the same stop cluster.
* **Modes** rank by generalised cost: *fastest* (time); *balanced* and *comfortable* weight waiting/walking
  1.25–1.5× and in-vehicle time by crowding (seats 1.0, standing 1.15–1.3, crowded 1.4–1.8), plus a transfer
  penalty of 2/5/10 min.
* **Alternatives**: rerun with the previous best option's routes banned; dedupe by route sequence.
* Each option: walk → board stop → route (towards) → wait → ride (stops, minutes, timetable minutes) → alight →
  transfers → walk; total, arrival, **range**, crowding per leg, GPS share, the **same trip at the quietest hour
  (06–22)**, a 24-hour trip-time profile, and ±15/30 min departures ("leave 15 min later: seats").
* Night: if no bus runs, it finds the first useful bus and the latest departure that still catches it.

## 6. Crowding layer (Problem 2)

1. Ticket route code (`8UP`) → GTFS route by normalised short name (98 % of tickets match exactly); else the
   base number with UP/DOWN stripped, picking the direction whose stops match the ticket.
2. Start/end stop **names** are fuzzy-matched (rapidfuzz WRatio ≥ 78, abbreviations expanded) **only within that
   route's own stops**, so "Ngr" ↔ "Nagar" and word order do not matter and a name cannot match a stop elsewhere.
   Missing end stop → the route's median ride length; end before start → the other direction.
3. Each ticket adds its passengers to every stretch it rides, at the half-hour it is there.
4. Per (route, stretch, day type, half-hour): passengers per day ÷ buses scheduled over that stretch in that
   half-hour × 1/0.035 (e-tickets ≈ 3–4 % of tickets) × 1.10 (pass holders and others not in ticket data) = people
   on board. Light smoothing over neighbouring half-hours.
5. Levels for a 12 m bus (35 seats + 35 standing): **seats available** < 32, **standing room** < 53, else
   **crowded**. Confidence from tickets in the window: ≥ 200 high, ≥ 50 medium, ≥ 5 low, else unknown.
6. Checks (CROWDING_CHECKS.md): morning/evening peaks, weekday vs weekend, busiest stretches, week-to-week
   correlation of route × hour boardings.

**Least reliable**: the absolute scale (e-ticket share varies by route, e.g. higher on AC e-buses); routes and hours
with few e-tickets; planned vs actual alighting; 9 m buses treated as 12 m. Rankings between options and hours are
more reliable than absolute numbers.

## 7. Limits

* About 10 % of stretches have no GPS observation (rare routes, stops the GPS seldom detects) and use the
  level 1–2 fallbacks; the app shows the share of each trip that is measured.
* Waiting uses expected waits, not real-time arrival predictions for a specific bus; the planner still
  under-estimates waits by ~3 min on average (it picks the route whose wait looks shortest).
* "Actual" waits and rides are measured from GPS stop visits, so buses whose GPS is off are invisible to both the
  model and its validation.
* Straight-line distances are used for speed sanity checks and the city-speed fallback (consistent with each
  other); shapes.txt is not needed for timing.
* Holidays are configured by hand (checked against the per-day profile).

## 8. Ready for Phase 2

The stretch-speed tables (by hour and day type) are a ready traffic input for **Problem 4** (EV feeder routes need
traffic-aware times on candidate roads), the `/matrix` endpoint gives travel-time matrices for **Problem 3**
OD work, and the ticket→stretch load pipeline extends to metro + bus segments for **Problem 5**.

## 9. Reproduce

`README.md` → Quick start. Every step prints and saves its match and drop rates.
