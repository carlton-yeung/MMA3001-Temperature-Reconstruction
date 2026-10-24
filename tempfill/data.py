"""Loading, auditing and regularising the NEMo environmental sensor data.

The raw export is a CSV (inside a ZIP) where each row holds one sensor upload
and the ``jsondata`` column contains every variable measured in that upload.
This module turns it into a clean, regularly spaced temperature series:

1. ``load_temperatures`` streams the CSV and extracts Temperature (°C) readings.
2. ``deduplicate`` collapses the heavily duplicated uploads (about 4 copies of
   every reading in this extract).
3. ``active_span`` discards stray readings separated from the main record by
   very long outages (e.g. a single reading in 2023 before a 636-day gap).
4. ``to_regular_steps`` places readings on a uniform 10-minute grid, with NaN
   wherever a reading is missing. These NaNs are what the project reconstructs.

Units: temperature in °C; time in Australia/Melbourne local time (tz-aware).
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = "Australia/Melbourne"
"""Sensors are in Clayton, Victoria. Raw timestamps are Unix seconds (UTC)."""
STEP_S = 600
"""Nominal sampling interval of the NEMo sensors (seconds)."""
T_MIN, T_MAX = 10.0, 40.0
"""Plausible indoor range (°C). Values outside are flagged, never silently deleted."""


# ----------------------------------------------------------------------------
# Raw loading
# ----------------------------------------------------------------------------
def extract_temperatures(json_text) -> list[tuple]:
    """Extract temperature readings from one ``jsondata`` cell.

    Args:
        json_text: JSON string holding a list of
            ``{"variable": {...}, "values": [{"time": ..., "value": ...}]}`` objects.

    Returns:
        List of ``(unix_time, value)`` tuples for the variable named
        ``"Temperature"`` with unit ``"°C"``. Malformed or missing JSON gives ``[]``.

    Example:
        >>> extract_temperatures('[{"variable": {"name": "Temperature", "unit": "°C"},'
        ...                      ' "values": [{"time": 0, "value": 21.5}]}]')
        [(0, 21.5)]
    """
    try:
        measurements = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(measurements, list):
        return []

    readings = []
    for m in measurements:
        var = m.get("variable") if isinstance(m, dict) else None
        # Check the unit as well as the name so different scales are never mixed.
        if isinstance(var, dict) and var.get("name") == "Temperature" and var.get("unit") == "°C":
            readings += [(r.get("time"), r.get("value"))
                         for r in m.get("values", []) if isinstance(r, dict)]
    return readings


def load_temperatures(zip_path: Path, chunksize: int = 20_000):
    """Load every temperature reading from the zipped sensor CSV.

    The CSV is streamed in chunks so the large ``jsondata`` strings are never
    all held in memory at once.

    Args:
        zip_path: ZIP archive containing one CSV with ``sensorid`` and
            ``jsondata`` columns.
        chunksize: Number of CSV rows parsed per chunk.

    Returns:
        ``(readings, raw_counts, stats)``:

        * ``readings`` - DataFrame with ``sensor_id`` (str), ``timestamp``
          (tz-aware, local) and ``temperature`` (°C); may contain duplicates.
        * ``raw_counts`` - ``Counter`` of CSV rows per sensor ID.
        * ``stats`` - dict of audit counts (rows read, rows without a
          temperature, invalid readings removed).

    Raises:
        FileNotFoundError: If ``zip_path`` does not exist.
        ValueError: If the archive has no CSV or no temperature readings.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Cannot find dataset at: {zip_path}")

    records, raw_counts = [], Counter()
    stats = {"csv_rows": 0, "rows_without_temperature": 0}

    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if csv_name is None:
            raise ValueError("No CSV file found in ZIP archive.")
        with zf.open(csv_name) as f:
            for chunk in pd.read_csv(f, chunksize=chunksize, dtype={"sensorid": str},
                                     usecols=["sensorid", "jsondata"]):
                stats["csv_rows"] += len(chunk)
                raw_counts.update(chunk["sensorid"].dropna())
                for sid, js in zip(chunk["sensorid"], chunk["jsondata"]):
                    readings = extract_temperatures(js) if isinstance(sid, str) else []
                    stats["rows_without_temperature"] += not readings
                    records += [(sid, t, v) for t, v in readings]

    df = pd.DataFrame(records, columns=["sensor_id", "timestamp", "temperature"])
    if df.empty:
        raise ValueError("No temperature measurements were extracted.")

    df["timestamp"] = (pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"),
                                      unit="s", utc=True).dt.tz_convert(LOCAL_TZ))
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    valid = df["timestamp"].notna() & np.isfinite(df["temperature"])
    stats["invalid_readings"] = int((~valid).sum())
    return df[valid].reset_index(drop=True), raw_counts, stats


def load_locations(xlsx_path: Path) -> dict[str, str]:
    """Map environmental sensor IDs to a readable location.

    Args:
        xlsx_path: The provided ``Sensor ID and Locations.xlsx`` workbook
            (sheet ``ENV``).

    Returns:
        Dict of sensor ID to ``"B<building> <room> <area>"``. Empty if the
        file is missing, so the pipeline still runs without it.
    """
    try:
        env = pd.read_excel(xlsx_path, sheet_name="ENV", dtype={"ID": str})
    except (FileNotFoundError, ValueError, ImportError):
        return {}
    return {str(sid): f"B{b} {room} {area}" for sid, b, room, area
            in zip(env["ID"], env["Building"], env["Floor/Room"], env["Area"])}


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def deduplicate(sensor_df: pd.DataFrame):
    """Collapse repeated timestamps for one sensor.

    Duplicates in this extract are identical re-uploads, so averaging is
    harmless; conflicting duplicates (same time, different value) are counted
    so this choice can be checked.

    Args:
        sensor_df: Readings for a single sensor (``timestamp``, ``temperature``).

    Returns:
        ``(clean, n_duplicates, n_conflicts)`` where ``clean`` is time-sorted
        with a boolean ``suspect`` column for values outside ``[T_MIN, T_MAX]``.
    """
    grouped = sensor_df.groupby("timestamp")["temperature"]
    n_conflicts = int((grouped.nunique() > 1).sum())
    clean = grouped.mean().reset_index()  # groupby output is already time-sorted
    clean["suspect"] = ~clean["temperature"].between(T_MIN, T_MAX)
    return clean, len(sensor_df) - len(clean), n_conflicts


def active_span(clean: pd.DataFrame, max_gap: pd.Timedelta | None = None):
    """Keep the longest stretch of a record that has no outage over ``max_gap``.

    Outages longer than ``max_gap`` are treated as out of scope (the sensor
    was offline, not briefly missing data), so reconstruction is only
    attempted within one active span.

    Args:
        clean: Time-sorted, duplicate-free readings.
        max_gap: Longest outage allowed inside a span (default 7 days).

    Returns:
        The rows of the span with the most readings (index reset).
    """
    max_gap = pd.Timedelta(days=7) if max_gap is None else max_gap
    span_id = (clean["timestamp"].diff() > max_gap).cumsum()
    longest = span_id.value_counts().idxmax()
    return clean[span_id == longest].reset_index(drop=True)


def _unix_seconds(ts: pd.Series) -> np.ndarray:
    """Convert a tz-aware datetime Series to integer Unix seconds."""
    return (ts.dt.tz_convert("UTC").dt.tz_localize(None)
            .to_numpy().astype("datetime64[s]").astype(np.int64))


def to_regular_steps(clean: pd.DataFrame, step_s: int = STEP_S, tz: str = LOCAL_TZ):
    """Place readings on a uniform grid of ``step_s`` slots, NaN where missing.

    Sensor clocks drift (the offset from the 10-minute mark wanders through
    the whole 10 minutes over months), so snapping to fixed clock times
    would merge or split readings. Instead each interval between consecutive
    readings is rounded to a whole number of steps:

        slot[i] = slot[i-1] + round((t[i] - t[i-1]) / step_s)

    This keeps the count of missing readings exact even under drift. The time
    of an empty slot is interpolated from its observed neighbours.

    Args:
        clean: Time-sorted, duplicate-free readings (``timestamp``,
            ``temperature``) from one active span.
        step_s: Nominal sampling interval in seconds.
        tz: Time zone for the output ``time`` column.

    Returns:
        ``(frame, stats)``. ``frame`` has one row per slot with columns
        ``time`` (tz-aware) and ``temperature`` (°C, NaN = missing).
        ``stats`` reports slots, observed, missing, collisions (readings less
        than half a step apart, averaged into one slot) and the largest
        timing error introduced by rounding (seconds).
    """
    if len(clean) < 2:
        raise ValueError("Need at least two readings to build a regular series.")
    secs = _unix_seconds(clean["timestamp"])
    dt = np.diff(secs)
    steps = np.rint(dt / step_s).astype(np.int64)
    slot = np.concatenate([[0], np.cumsum(steps)])

    per_slot = (pd.DataFrame({"slot": slot, "secs": secs,
                              "temperature": clean["temperature"].to_numpy()})
                .groupby("slot").mean())
    n = int(slot[-1]) + 1
    temperature = np.full(n, np.nan)
    temperature[per_slot.index] = per_slot["temperature"]
    t_slot = np.interp(np.arange(n), per_slot.index, per_slot["secs"])

    frame = pd.DataFrame({
        "time": pd.to_datetime(np.rint(t_slot).astype(np.int64), unit="s", utc=True)
                  .tz_convert(tz),
        "temperature": temperature,
    })
    observed = int(np.isfinite(temperature).sum())
    stats = {
        "slots": n, "observed": observed, "missing": n - observed,
        "collisions": int((steps == 0).sum()),
        "max_timing_error_s": float(np.abs(dt - steps * step_s).max()),
    }
    return frame, stats


def gap_runs(missing: np.ndarray):
    """Find runs of consecutive ``True`` values.

    Args:
        missing: Boolean array, ``True`` where a reading is missing.

    Returns:
        ``(starts, lengths)`` integer arrays, one entry per run.

    Example:
        >>> gap_runs(np.array([0, 1, 1, 0, 1], dtype=bool))
        (array([1, 4]), array([2, 1]))
    """
    edges = np.diff(np.concatenate([[0], np.asarray(missing, dtype=np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


# ----------------------------------------------------------------------------
# High-level helpers
# ----------------------------------------------------------------------------
def audit_sensors(readings: pd.DataFrame, raw_counts: Counter,
                  locations: dict | None = None) -> pd.DataFrame:
    """Build a data-quality table with one row per sensor.

    Args:
        readings: Output of ``load_temperatures``.
        raw_counts: CSV rows per sensor from ``load_temperatures``.
        locations: Optional sensor ID to location mapping.

    Returns:
        DataFrame sorted by number of unique readings (descending).
    """
    locations = locations or {}
    rows = []
    for sid, group in readings.groupby("sensor_id"):
        clean, n_dup, n_conf = deduplicate(group)
        gaps_min = clean["timestamp"].diff().dt.total_seconds() / 60
        rows.append({
            "sensor_id": sid,
            "location": locations.get(sid, "not in location table"),
            "csv_rows": raw_counts.get(sid, 0),
            "unique_readings": len(clean),
            "duplicates": n_dup,
            "conflicting_duplicates": n_conf,
            "start": clean["timestamp"].min(),
            "end": clean["timestamp"].max(),
            "median_interval_min": gaps_min.median(),
            "gaps_over_15min": int((gaps_min > 15).sum()),
            "gaps_over_1day": int((gaps_min > 1440).sum()),
            "suspect_readings": int(clean["suspect"].sum()),
            "t_min": clean["temperature"].min(),
            "t_max": clean["temperature"].max(),
        })
    return pd.DataFrame(rows).sort_values("unique_readings", ascending=False,
                                          ignore_index=True)


def prepare_sensor(readings: pd.DataFrame, sensor_id: str,
                   max_gap: pd.Timedelta | None = None):
    """Run the full cleaning chain for one sensor.

    Args:
        readings: Output of ``load_temperatures``.
        sensor_id: Sensor to prepare.
        max_gap: Outage length that splits the record into spans (default 7 days).

    Returns:
        ``(frame, stats)`` from ``to_regular_steps`` for the sensor's longest
        active span, with deduplication counts added to ``stats``.

    Raises:
        KeyError: If the sensor has no readings.
    """
    group = readings[readings["sensor_id"] == sensor_id]
    if group.empty:
        raise KeyError(f"No readings for sensor {sensor_id!r}.")
    clean, n_dup, n_conf = deduplicate(group)
    span = active_span(clean, max_gap)
    frame, stats = to_regular_steps(span)
    stats.update(duplicates=n_dup, conflicting_duplicates=n_conf,
                 dropped_outside_span=len(clean) - len(span),
                 suspect=int(span["suspect"].sum()),
                 start=str(frame["time"].iloc[0]), end=str(frame["time"].iloc[-1]))
    return frame, stats


def train_test_split(frame: pd.DataFrame, train_fraction: float = 0.7) -> int:
    """Return the slot index splitting a series chronologically.

    Slots before the index are used for training and tuning; slots after are
    held out for testing. A chronological split avoids leaking future
    behaviour into model selection.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    return int(len(frame) * train_fraction)
