"""Tests for loading, cleaning and regularising sensor data."""
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from tempfill.data import (active_span, deduplicate, extract_temperatures, gap_runs,
                           load_temperatures, to_regular_steps, train_test_split)


def _json(t, value, name="Temperature", unit="°C"):
    return json.dumps([
        {"values": [{"time": t, "value": 4.1}], "variable": {"name": "Battery", "unit": "V"}},
        {"values": [{"time": t, "value": value}], "variable": {"name": name, "unit": unit}},
    ])


def _ts(seconds):
    return pd.Series(pd.to_datetime(seconds, unit="s", utc=True)).dt.tz_convert("Australia/Melbourne")


class TestExtract:
    def test_finds_temperature_only(self):
        assert extract_temperatures(_json(100, 21.5)) == [(100, 21.5)]

    def test_rejects_wrong_unit(self):
        assert extract_temperatures(_json(100, 70.0, unit="°F")) == []

    @pytest.mark.parametrize("bad", [None, "", "not json", "{}", "[1, 2]", float("nan")])
    def test_malformed_input_returns_empty(self, bad):
        assert extract_temperatures(bad) == []


def test_load_temperatures_from_zip(tmp_path):
    csv = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "sensorid": ["A", "A", "B", None],
        "jsondata": [_json(0, 20.0), _json(0, 20.0), _json(600, 22.0), _json(0, 1.0)],
    })
    zpath = tmp_path / "data.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("data.csv", csv.to_csv(index=False))
    readings, counts, stats = load_temperatures(zpath)
    assert stats["csv_rows"] == 4
    assert stats["rows_without_temperature"] == 1   # row with missing sensor ID
    assert counts == {"A": 2, "B": 1}
    assert len(readings) == 3
    assert str(readings["timestamp"].dt.tz) == "Australia/Melbourne"


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_temperatures(tmp_path / "nope.zip")


def test_deduplicate_counts_and_conflicts():
    df = pd.DataFrame({"timestamp": _ts([0, 0, 600, 600, 1200]),
                       "temperature": [20.0, 20.0, 21.0, 23.0, 50.0]})
    clean, n_dup, n_conf = deduplicate(df)
    assert len(clean) == 3 and n_dup == 2 and n_conf == 1
    assert clean["temperature"].iloc[1] == pytest.approx(22.0)   # averaged
    assert clean["suspect"].tolist() == [False, False, True]     # 50 °C flagged


def test_active_span_keeps_longest_stretch():
    secs = [0] + list(range(100 * 86400, 100 * 86400 + 600 * 10, 600))
    clean = pd.DataFrame({"timestamp": _ts(secs), "temperature": 20.0})
    span = active_span(clean)
    assert len(span) == 10                 # the isolated first reading is dropped


class TestRegularSteps:
    def test_gaps_become_nan(self):
        secs = np.array([0, 600, 1200, 3000, 3600])       # 1800 -> 3000 skips 2 slots
        clean = pd.DataFrame({"timestamp": _ts(secs), "temperature": [1.0, 2, 3, 6, 7]})
        frame, stats = to_regular_steps(clean)
        assert stats["slots"] == 7 and stats["missing"] == 2
        assert np.isnan(frame["temperature"].iloc[[3, 4]]).all()
        assert frame["temperature"].iloc[5] == 6.0

    def test_robust_to_clock_drift(self):
        # Interval of 610 s drifts 10 s per reading: after 60 readings the
        # offset has wandered a full 10 minutes, but no slot is lost or merged.
        secs = np.arange(200) * 610
        clean = pd.DataFrame({"timestamp": _ts(secs), "temperature": np.arange(200.0)})
        frame, stats = to_regular_steps(clean)
        assert stats["missing"] == 0 and stats["collisions"] == 0
        assert len(frame) == 200

    def test_collision_is_averaged(self):
        secs = np.array([0, 600, 700, 1200])               # 600 and 700 share a slot
        clean = pd.DataFrame({"timestamp": _ts(secs), "temperature": [1.0, 2.0, 4.0, 5.0]})
        frame, stats = to_regular_steps(clean)
        assert stats["collisions"] == 1
        assert frame["temperature"].tolist() == [1.0, 3.0, 5.0]


def test_gap_runs():
    starts, lengths = gap_runs(np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool))
    assert starts.tolist() == [0, 4, 6] and lengths.tolist() == [2, 1, 3]


def test_train_test_split_bounds():
    frame = pd.DataFrame({"temperature": np.zeros(100)})
    assert train_test_split(frame, 0.7) == 70
    with pytest.raises(ValueError):
        train_test_split(frame, 1.2)
