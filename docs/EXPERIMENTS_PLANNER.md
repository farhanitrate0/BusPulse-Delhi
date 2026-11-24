# Planner corrections — experiments that did not make it

Chosen on training days only: corrections fitted on **4–17 Oct** (2,800 random trips → 2,448 bus legs with real
rides), scored on **18–24 Oct** (1,400 trips → 1,283 legs). The held-out week 25–31 Oct was not used.
Scripts: `scripts/13_ride_correction.py`, `scripts/14_wait_correction.py`.

## Whole-ride correction

Real rides take longer than the sum of their stop-to-stop stretches. v5 multiplies the stretch sum by one median
ratio per time of day. Could the ratio also depend on ride length and on the route?

| correction | MAE min | bias min | MAPE % | median APE % | within 20 % |
|---|---|---|---|---|---|
| none (stretch model alone) | 5.18 | −4.01 | 16.82 | 12.72 | 68.9 |
| **per time of day (v5, kept)** | **4.65** | −1.52 | 17.03 | 13.54 | 68.5 |
| per time of day × ride length | 4.66 | −1.32 | 16.89 | 13.36 | 68.8 |
| + per route (shrunk, K = 5) | 4.66 | −0.91 | 17.12 | 13.64 | 67.7 |
| + per route × time of day | 4.65 | −1.11 | 16.87 | 13.35 | 69.3 |

Longer rides do need a slightly bigger factor (evening peak 1.12 for short rides → 1.21 for rides over 35 min),
and per-route terms remove some bias. But the error does not move: what is left is the spread between
individual buses on the same route and hour, which history alone cannot predict. **Kept v5.**

## Wait correction

v5 scales the GPS headway wait by one factor (real ÷ predicted, ratio of means) and the timetable half-headway by
another. Per-time-of-day factors:

| correction | MAE min | bias min | RMSE min |
|---|---|---|---|
| none | 10.67 | −6.92 | 19.60 |
| **one factor per source (v5, kept)** | 11.01 | −2.58 | 18.17 |
| one factor per source × time of day | 10.97 | −2.41 | 17.95 |
| same, shrunk towards the global factor | 11.01 | −2.43 | 18.04 |

Waits have an 18-minute RMSE whatever the correction: Delhi buses bunch and skip, so the wait for the next bus is
close to random given the history. Only a live feed of where the buses are now can cut it.
**Kept v5.**

## Takeaway

With history alone the planner is at its noise floor: corrections can shift the average but not the
trip-to-trip error. The next real gain is live data. The live interface is built and tested on a replayed
day, so connecting the city's GTFS-realtime feed (otd.delhi.gov.in) is the next step.
