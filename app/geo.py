"""Small geo helpers. A local equirectangular projection is accurate to <0.5 % inside Delhi."""
from __future__ import annotations

import numpy as np

LAT0, LON0 = 28.6, 77.1
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0 * np.cos(np.radians(LAT0))


def to_xy(lat, lon):
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    return (lon - LON0) * M_PER_DEG_LON, (lat - LAT0) * M_PER_DEG_LAT


def to_latlon(x, y):
    return np.asarray(y) / M_PER_DEG_LAT + LAT0, np.asarray(x) / M_PER_DEG_LON + LON0


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000.0 * np.arcsin(np.sqrt(a))
