"""tempfill - reconstruction of missing indoor temperature readings.

MMA3001 Numerical Methods and Machine Learning, individual project
(Monash Smart Infrastructure environmental sensor data, Dataset 2).

Problem
-------
NEMo environmental sensors in Buildings 60 and 69 report temperature every
10 minutes, but readings go missing: about 11% of slots for the main sensor,
mostly single dropped uploads plus occasional multi-day outages. Comfort,
HVAC and energy analyses need continuous records, so the gaps must be
filled, and the error of the filled values must be known.

Inputs and outputs
------------------
* **Input:** the zipped NEMo CSV export (``sensorid``, ``jsondata``) plus
  the optional sensor-location workbook. Temperature in °C, timestamps in
  Unix seconds (UTC).
* **Output:** for each usable sensor, a continuous 10-minute series (local
  time) where every missing slot is filled, flagged as reconstructed, and
  tagged with its validated expected absolute error (°C).

Modules
-------
* ``tempfill.data`` - loading, deduplication, regular 10-minute grid.
* ``tempfill.methods`` - the gap-filling methods.
* ``tempfill.validation`` - synthetic-gap validation and metrics.
* ``tempfill.performance`` - timing, memory, scaling, solver stability.
* ``tempfill.experiments`` - the full experimental protocol.
* ``tempfill.plots`` - report figures.

Run ``python main.py`` from the repository root to reproduce all results.
"""

__version__ = "1.0.0"
