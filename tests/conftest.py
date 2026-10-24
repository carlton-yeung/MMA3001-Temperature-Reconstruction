"""Shared test fixtures: small synthetic series with known answers."""
import numpy as np
import pandas as pd
import pytest


def make_frame(values, start="2025-03-03", tz="Australia/Melbourne"):
    """Wrap an array of temperatures (NaN = missing) as a 10-minute frame."""
    values = np.asarray(values, dtype=float)
    time = pd.Series(pd.date_range(start, periods=len(values), freq="600s", tz=tz))
    return pd.DataFrame({"time": time, "temperature": values})


@pytest.fixture
def daily_frame():
    """Two weeks of an exact daily + weekly sinusoid (no noise, no gaps)."""
    n = 14 * 144
    f = make_frame(np.zeros(n))
    hour = (f["time"].dt.hour + f["time"].dt.minute / 60).to_numpy()
    week_hour = f["time"].dt.dayofweek.to_numpy() * 24 + hour
    f["temperature"] = (21 + 1.5 * np.cos(2 * np.pi * hour / 24)
                        + 0.5 * np.sin(4 * np.pi * hour / 24)
                        + 0.3 * np.cos(2 * np.pi * week_hour / 168))
    return f
