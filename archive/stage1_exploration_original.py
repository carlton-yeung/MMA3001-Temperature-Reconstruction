# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 22:17:26 2026

@author: yeung
"""
"""
copy this in terminal:
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib scipy pytest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
from pathlib import Path
import zipfile
import json

"""
MMA3001 - Temperature Reconstruction
Stage 1: Data Exploration and Preparation

Project:
Reconstruction of Missing Indoor Temperature Sensor Data
Using Numerical Interpolation Methods.

Objectives:
1. Load the environmental sensor dataset.
2. Identify all available environmental sensors.
3. Extract temperature measurements from each sensor.
4. Assess data quality and sampling consistency.
5. Compare sensors using numerical statistics.
6. Generate and save temperature graphs.
7. Select a sensor for further analysis.


"""
# ==========================================================
# STEP 1: LOCATE THE DATASET
# ==========================================================

project_folder = Path(__file__).resolve().parent
results_folder = project_folder / "results"
results_folder.mkdir(exist_ok=True)

# Construct the location of our environmental dataset.
data_path = (
    project_folder
    / "data"
    / "5EnvSensor_MayToDec2024_180kRows.zip"
)
if not data_path.is_file():
    raise FileNotFoundError(
        f"Cannot find dataset at: {data_path}"
    )
print("STEP 1: Dataset successfully located")

selected_sensor_id = None
minimum_temperature = 10
maximum_temperature = 40

# ==========================================================
# STEP 2: LOAD AND INSPECT THE RAW DATASET
# ==========================================================

print("\nSTEP 2: Inspecting environmental measures ")

# Store extracted measurements from all sensors.
temperature_records = []

# Record the number of raw CSV entries for each sensor.
raw_sensor_counts = {}

# Track malformed JSON and invalid measurement objects.
invalid_json_count = 0

# Count all rows in the original CSV.
total_rows = 0

# Open the ZIP without manually extracting its contents.
with zipfile.ZipFile(data_path) as archive:

    # Identify CSV files within the archive.
    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise ValueError("No CSV file found in ZIP archive.")

    # Open the first CSV file.
    with archive.open(csv_files[0]) as file:

        # Process 5000 records at a time.
        # Chunking controls memory consumption.
        for chunk in pd.read_csv(
            file,
            chunksize=5000,
            dtype={"sensorid": str},
            usecols=["sensorid", "jsondata"]
        ):

            # Update the number of processed CSV rows.
            total_rows += len(chunk)

            # Count records belonging to each sensor.
            counts = chunk["sensorid"].value_counts()

            for sensor, count in counts.items():
                raw_sensor_counts[sensor] = (
                    raw_sensor_counts.get(sensor, 0) + count
                )

            # Extract measurements from every sensor.
            for sensor, json_text in zip(
                chunk["sensorid"], chunk["jsondata"]
            ):

                # Skip missing sensor IDs or JSON content.
                if pd.isna(sensor) or pd.isna(json_text):
                    invalid_json_count += 1
                    continue

                try:
                    # Convert JSON text into Python objects.
                    measurements = json.loads(json_text)

                except (json.JSONDecodeError, TypeError):
                    invalid_json_count += 1
                    continue

                if not isinstance(measurements, list):
                    invalid_json_count += 1
                    continue

                # Each JSON record contains different variables,
                # such as humidity, CO2 and temperature.
                for measurement in measurements:

                    if not isinstance(measurement, dict):
                        continue

                    variable = measurement.get("variable", {})

                    if not isinstance(variable, dict):
                        continue

                    # Identify the temperature variable.
                    # Check units to avoid mixing measurements.
                    if (
                        variable.get("name") == "Temperature"
                        and variable.get("unit") == "°C"
                    ):

                        # A variable may contain multiple readings.
                        for reading in measurement.get(
                            "values", []
                        ):

                            if not isinstance(reading, dict):
                                continue

                            # Store the sensor, timestamp and value.
                            temperature_records.append({
                                "sensor_id": sensor,
                                "timestamp": reading.get("time"),
                                "temperature": reading.get("value")
                            })

print("Raw CSV rows:", total_rows)
print("Invalid JSON or missing ID rows:", invalid_json_count)

print("\nEnvironmental sensor IDs:")
for sensor, count in sorted(raw_sensor_counts.items()):
    print(f"{sensor}: {count} CSV records")

    # ==========================================================
# STEP 3: CONVERT AND VALIDATE THE MEASUREMENTS
# ==========================================================

print("\nSTEP 3: Converting temperature measurements")

# Convert the extracted list into a pandas DataFrame.
temperature_df = pd.DataFrame(temperature_records)

if temperature_df.empty:
    raise ValueError("No temperature measurements were extracted.")

# Convert Unix seconds into UTC timestamps.
# UTC is used consistently to avoid timezone ambiguity.
temperature_df["timestamp"] = pd.to_datetime(
    temperature_df["timestamp"],
    unit="s",
    utc=True,
    errors="coerce"
)

# Ensure that all temperature measurements are numeric.
temperature_df["temperature"] = pd.to_numeric(
    temperature_df["temperature"],
    errors="coerce"
)

# Count missing or invalid readings before cleaning.
invalid_rows = (
    temperature_df["timestamp"].isna()
    | ~np.isfinite(temperature_df["temperature"])
)

print("Invalid timestamp or temperature rows:",
      invalid_rows.sum())

# Exclude invalid measurements from the usable dataset.
temperature_df = temperature_df.loc[
    ~invalid_rows
].copy()

# Identify potentially suspicious temperatures.
# We flag rather than automatically delete them.
temperature_df["suspect_temperature"] = (
    (temperature_df["temperature"] < minimum_temperature)
    | (temperature_df["temperature"] > maximum_temperature)
)

# ==========================================================
# STEP 4: COMPARE DATA QUALITY ACROSS ALL SENSORS
# ==========================================================

print("\nSTEP 4: Comparing environmental sensors...")

# Create an empty list to store statistics for each sensor.
sensor_summaries = []

# Create a dictionary to store each sensor's cleaned dataset.
sensor_datasets = {}

# Group the temperature measurements by sensor ID.
# This allows us to analyse every sensor individually.
for sensor, sensor_data in temperature_df.groupby("sensor_id"):

    # Sort observations chronologically.
    sensor_data = sensor_data.sort_values(
        "timestamp"
    ).copy()

    # ------------------------------------------------------
    # 4.1: IDENTIFY DUPLICATE MEASUREMENTS
    # ------------------------------------------------------

    # Count records with repeated timestamps.
    # The first occurrence is not counted as a duplicate.
    duplicate_count = sensor_data.duplicated(
        subset=["timestamp"]
    ).sum()

    # Count timestamps containing different temperature values.
    # Conflicting duplicates may indicate data quality issues.
    conflicting_timestamps = (
        sensor_data.groupby("timestamp")["temperature"]
        .nunique()
        .gt(1)
        .sum()
    )

    # ------------------------------------------------------
    # 4.2: CONSOLIDATE DUPLICATE TIMESTAMPS
    # ------------------------------------------------------

    # Combine temperature measurements sharing a timestamp.
    # For now, we take the mean of their values.
    # This is a provisional cleaning decision that we
    # will investigate before reconstruction experiments.
    clean_data = (
        sensor_data
        .groupby("timestamp", as_index=False)
        .agg(
            temperature=("temperature", "mean"),
            suspect=("suspect_temperature", "max")
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # 4.3: CALCULATE SAMPLING INTERVALS
    # ------------------------------------------------------

    # Calculate the time difference between readings.
    # The first observation has NaN because it does not
    # have a previous measurement.
    clean_data["gap_minutes"] = (
        clean_data["timestamp"]
        .diff()
        .dt.total_seconds() / 60
    )

    # Remove missing intervals before calculating statistics.
    valid_intervals = (
        clean_data["gap_minutes"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # Only calculate the median when valid intervals exist.
    # This prevents the "Mean of empty slice" warning.
    if not valid_intervals.empty:

        median_interval = valid_intervals.median()

    else:

        # If only one unique timestamp exists, there
        # is no sampling interval to calculate.
        median_interval = np.nan

        print(
            f"Sensor {sensor}: insufficient timestamps "
            "to calculate sampling interval."
        )

    # ------------------------------------------------------
    # 4.4: IDENTIFY LARGE TIME GAPS
    # ------------------------------------------------------

    # Define large gaps as intervals exceeding 1.5 times
    # the sensor's typical sampling interval.
    # This is an exploratory threshold, not a verified
    # definition of a missing measurement.
    if pd.notna(median_interval) and median_interval > 0:

        gap_threshold = 1.5 * median_interval

        # True indicates an unusually long interval.
        large_gap_mask = (
            clean_data["gap_minutes"] > gap_threshold
        )

        # Each large gap marks the beginning of a new
        # continuous segment of temperature measurements.
        segment_number = large_gap_mask.cumsum()

        # Count measurements in each continuous segment.
        segment_sizes = clean_data.groupby(
            segment_number
        ).size()

        # Record the longest continuous sequence.
        longest_segment = int(segment_sizes.max())

        # Count all unusually large gaps.
        large_gap_count = int(large_gap_mask.sum())

    else:

        # If the sampling interval cannot be determined,
        # large gaps cannot be assessed reliably.
        large_gap_count = np.nan

        # A single reading forms a segment of length one.
        longest_segment = len(clean_data)

    # ------------------------------------------------------
    # 4.5: CALCULATE SENSOR SUMMARY STATISTICS
    # ------------------------------------------------------

    # Format timestamps for display.
    # Original timestamps remain timezone-aware.
    start_time = (
        clean_data["timestamp"]
        .min()
        .strftime("%Y-%m-%d %H:%M:%S")
    )

    end_time = (
        clean_data["timestamp"]
        .max()
        .strftime("%Y-%m-%d %H:%M:%S")
    )

    # Save each sensor's measurements and quality statistics.
    sensor_summaries.append({
        "Sensor ID": sensor,
        "Unique readings": len(clean_data),
        "Start (UTC)": start_time,
        "End (UTC)": end_time,
        "Median interval (min)": median_interval,
        "Large gaps": large_gap_count,
        "Longest continuous run": longest_segment,
        "Duplicate entries": int(duplicate_count),
        "Conflicting timestamps": int(
            conflicting_timestamps
        ),
        "Suspect readings": int(
            clean_data["suspect"].sum()
        ),
        "Min temp (°C)": clean_data["temperature"].min(),
        "Max temp (°C)": clean_data["temperature"].max()
    })

    # Store the cleaned measurements for each sensor.
    # We will use these later for plotting and selection.
    sensor_datasets[sensor] = clean_data


# ----------------------------------------------------------
# 4.6: CREATE THE SENSOR COMPARISON TABLE
# ----------------------------------------------------------

# Convert the sensor statistics into a pandas DataFrame.
comparison_df = pd.DataFrame(sensor_summaries)

# Sort by longest continuous run, from largest to smallest.
# This helps identify sensors with suitable data coverage,
# but does not automatically determine the best sensor.
comparison_df = comparison_df.sort_values(
    "Longest continuous run",
    ascending=False
)

# Adjust pandas settings to display the complete table.
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# Print the results.
print("\nSENSOR COMPARISON:")
print(comparison_df.to_string(index=False))





# ==========================================================
# STEP 6: VISUALISE AND SAVE TEMPERATURE GRAPHS
# ==========================================================

print("\nSTEP 6: Generating temperature graphs...")

# Create one figure with a separate subplot for each sensor.
# This allows all sensors to be compared in one window.
number_of_sensors = len(sensor_datasets)

fig, axes = plt.subplots(
    nrows=number_of_sensors,
    ncols=1,
    figsize=(14, 3.5 * number_of_sensors)
)

# Convert axes into an array so the code also works
# when the dataset contains only one sensor.
axes = np.atleast_1d(axes)

# ----------------------------------------------------------
# PLOT TEMPERATURE MEASUREMENTS FOR EACH SENSOR
# ----------------------------------------------------------

# Loop through sensor data and its corresponding subplot.
for ax, (sensor_id, clean_data) in zip(
    axes, sensor_datasets.items()
):

    # Count the number of unique temperature measurements.
    number_of_readings = len(clean_data)

    if number_of_readings < 3:

        # Sensors with one or two readings are shown as points.
        # Joining these points would imply a continuous trend
        # that is not supported by the available measurements.
        ax.scatter(
            clean_data["timestamp"],
            clean_data["temperature"],
            color="steelblue",
            s=40,
            label="Recorded temperature"
        )

    else:

        # --------------------------------------------------
        # IDENTIFY LARGE GAPS BETWEEN MEASUREMENTS
        # --------------------------------------------------

        # Calculate the time difference between consecutive
        # measurements, expressed in minutes.
        time_gaps = (
            clean_data["timestamp"]
            .diff()
            .dt.total_seconds() / 60
        )

        # Calculate the typical sampling interval.
        # The median is used because it is less affected
        # by unusually large gaps.
        valid_gaps = time_gaps.dropna()

        if not valid_gaps.empty:

            typical_interval = valid_gaps.median()

            # A gap greater than 1.5 times the typical
            # interval is treated as an interruption.
            if typical_interval > 0:
                gap_threshold = 1.5 * typical_interval
            else:
                gap_threshold = 15

        else:
            gap_threshold = 15

        # Identify measurements that follow large gaps.
        large_gap_mask = time_gaps > gap_threshold

        # Every large gap starts a new continuous segment.
        # cumsum() assigns increasing segment numbers.
        segment_id = large_gap_mask.cumsum()

        # --------------------------------------------------
        # PLOT EACH CONTINUOUS SEGMENT SEPARATELY
        # --------------------------------------------------

        # Plotting segments separately prevents Matplotlib
        # from connecting measurements across long gaps.
        for _, segment in clean_data.groupby(segment_id):

            if len(segment) == 1:

                # A single observation is displayed as a dot.
                ax.scatter(
                    segment["timestamp"],
                    segment["temperature"],
                    color="steelblue",
                    s=8
                )

            else:

                # Join observations only within a segment.
                ax.plot(
                    segment["timestamp"],
                    segment["temperature"],
                    color="steelblue",
                    linewidth=0.8
                )

    # ------------------------------------------------------
    # FORMAT THE CURRENT SENSOR GRAPH
    # ------------------------------------------------------

    # Show the sensor ID and number of unique readings.
    ax.set_title(
        f"Sensor {sensor_id} - "
        f"{number_of_readings} unique readings"
    )

    # Label axes with measurement units.
    ax.set_xlabel("Date and Time (UTC)")
    ax.set_ylabel("Temperature (°C)")

    # Add a grid to make measurements easier to compare.
    ax.grid(True, alpha=0.3)

    # Display a legend for sensors with very few readings.
    if number_of_readings < 3:
        ax.legend()

# ----------------------------------------------------------
# SAVE AND DISPLAY THE COMBINED FIGURE
# ----------------------------------------------------------

# Adjust spacing between subplots to prevent overlap.
fig.tight_layout()

# Specify the location for the combined graph.
output_path = (
    results_folder / "all_sensor_temperatures.png"
)

# Save the figure at a resolution suitable for reports.
fig.savefig(
    output_path,
    dpi=300
)

print("\nCombined graph saved to:", output_path)

# Display all five sensor graphs in one window.
plt.show()

# Close the figure after the window is closed.
plt.close(fig)

print("\nSTEP 6 COMPLETED")


# ==========================================================
# STEP 7: SELECT SENSOR AND IDENTIFY CONTINUOUS PERIODS
# ==========================================================

print("\nSTEP 7: Selecting sensor and continuous periods...")

# ----------------------------------------------------------
# 7.1: SELECT A SENSOR
# ----------------------------------------------------------

# Display all sensors available for selection.
# The sensor comparison table was generated in Step 4.
print("\nAvailable sensor IDs:")

for sensor_id in comparison_df["Sensor ID"]:
    print(sensor_id)

# Ask the user to enter a sensor ID.
# This avoids hard-coding the selection into our program.
selected_sensor_id = input(
    "\nEnter the sensor ID you want to analyse: "
).strip()

# Verify that the selected sensor exists in our dataset.
if selected_sensor_id not in sensor_datasets:
    raise ValueError(
        f"Invalid sensor ID: {selected_sensor_id}"
    )

# Retrieve the cleaned temperature measurements.
# .copy() prevents changes to the original dataset.
selected_data = sensor_datasets[
    selected_sensor_id
].copy()

# A sensor needs at least two readings to have a time interval.
if len(selected_data) < 2:
    raise ValueError(
        "Selected sensor has insufficient temperature readings."
    )

print("\nSelected sensor:", selected_sensor_id)
print("Total unique readings:", len(selected_data))


# ----------------------------------------------------------
# 7.2: IDENTIFY THE TYPICAL SAMPLING INTERVAL
# ----------------------------------------------------------

# Calculate the time difference between consecutive readings.
# Convert the difference from seconds to minutes.
selected_data["gap_minutes"] = (
    selected_data["timestamp"]
    .diff()
    .dt.total_seconds() / 60
)

# Remove missing, zero or negative intervals.
# The first timestamp has no preceding measurement.
valid_intervals = selected_data.loc[
    selected_data["gap_minutes"] > 0,
    "gap_minutes"
]

# Check that at least one valid interval exists.
if valid_intervals.empty:
    raise ValueError(
        "Cannot determine the sensor's sampling interval."
    )

# Calculate the median sampling interval.
# The median reduces the influence of unusually large gaps.
typical_interval = valid_intervals.median()

print("\nTypical sampling interval:",
      typical_interval, "minutes")


# ----------------------------------------------------------
# 7.3: IDENTIFY CONTINUOUS PERIODS
# ----------------------------------------------------------

# Allow a 50% tolerance above the typical interval.
# For a 10-minute interval, gaps greater than
# 15 minutes start a new continuous period.
#
# This is an exploratory engineering assumption.
gap_threshold = 1.5 * typical_interval

print("Continuous-period gap threshold:",
      gap_threshold, "minutes")

# Identify readings that follow a large time gap.
large_gap_mask = (
    selected_data["gap_minutes"] > gap_threshold
)

# Assign each uninterrupted sequence a segment number.
# Every large gap begins a new segment.
selected_data["segment_id"] = (
    large_gap_mask.cumsum() + 1
)


# ----------------------------------------------------------
# 7.4: CALCULATE STATISTICS FOR EACH PERIOD
# ----------------------------------------------------------

# Group measurements by their continuous-period ID.
segment_summary = (
    selected_data
    .groupby("segment_id")
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        readings=("temperature", "size"),
        min_temperature=("temperature", "min"),
        max_temperature=("temperature", "max")
    )
    .reset_index()
)

# Calculate each period's actual duration in hours.
segment_summary["duration_hours"] = (
    segment_summary["end_time"]
    - segment_summary["start_time"]
).dt.total_seconds() / 3600

# Sort periods from most readings to fewest.
# Longer periods give us more flexibility when
# testing different missing-data durations.
segment_summary = segment_summary.sort_values(
    "readings",
    ascending=False
).reset_index(drop=True)

print("\nTop 10 longest continuous periods:")

print(
    segment_summary.head(10).to_string(index=False)
)


# ----------------------------------------------------------
# 7.5: SELECT SUITABLE PERIODS
# ----------------------------------------------------------

# Require at least 24 measurements per period.
# At a 10-minute interval, 24 measurements span
# approximately 3 hours and 50 minutes.
#
# This is a provisional minimum for our experiments.
# We may change it depending on the gap sizes tested.
minimum_readings = 24

# Keep periods containing sufficient measurements.
suitable_segments = segment_summary[
    segment_summary["readings"] >= minimum_readings
].copy()

print("\nNumber of suitable continuous periods:",
      len(suitable_segments))

# Stop if no period satisfies the minimum requirement.
if suitable_segments.empty:
    raise ValueError(
        "No continuous periods meet the minimum length."
    )


# ----------------------------------------------------------
# 7.6: SAVE CONTINUOUS PERIOD INFORMATION
# ----------------------------------------------------------

# Save the summary of all identified periods.
summary_path = (
    results_folder / "continuous_period_summary.csv"
)

segment_summary.to_csv(
    summary_path,
    index=False
)

# Identify the IDs of suitable periods.
suitable_ids = suitable_segments["segment_id"]

# Extract all measurements belonging to suitable periods.
suitable_data = selected_data[
    selected_data["segment_id"].isin(suitable_ids)
].copy()

# Save the measurements needed for Stage 2.
output_path = (
    results_folder / "prepared_temperature_data.csv"
)

suitable_data.to_csv(
    output_path,
    index=False
)

print("\nContinuous period summary saved to:",
      summary_path)

print("Prepared temperature data saved to:",
      output_path)


# ----------------------------------------------------------
# 7.7: VISUALISE THE LONGEST CONTINUOUS PERIOD
# ----------------------------------------------------------

# Select the ID of the period with the most measurements.
longest_segment_id = segment_summary.iloc[0]["segment_id"]

# Retrieve measurements belonging to that period.
longest_segment = selected_data[
    selected_data["segment_id"] == longest_segment_id
].copy()

# Create a figure showing temperature over this period.
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    longest_segment["timestamp"],
    longest_segment["temperature"],
    color="steelblue",
    marker="o",
    markersize=3,
    linewidth=1,
    label="Recorded temperature"
)

# Add a descriptive title and axis labels.
ax.set_title(
    f"Longest Continuous Period - Sensor {selected_sensor_id}"
)

ax.set_xlabel("Date and Time (UTC)")
ax.set_ylabel("Temperature (°C)")
# Import Matplotlib's date formatting tools.
import matplotlib.dates as mdates

# Format the x-axis as Month-Day Hour:Minute.
# Example: 03-04 22:00
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%m-%d %H:%M", tz=mdates.UTC)
)

# Place a major tick every hour.
ax.xaxis.set_major_locator(
    mdates.HourLocator(interval=1, tz=mdates.UTC)
)

# Rotate labels slightly to prevent overlapping.
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

# Save the graph for documentation.
graph_path = (
    results_folder / "longest_continuous_period.png"
)

fig.savefig(
    graph_path,
    dpi=300
)

# Display the graph in VS Code.
plt.show()

# Close the figure to release memory.
plt.close(fig)

print("\nSTEP 7 COMPLETED!")
print("STAGE 1: DATA EXPLORATION AND PREPARATION COMPLETED!")
