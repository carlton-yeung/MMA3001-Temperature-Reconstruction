# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 22:17:26 2026

@author: yeung
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator

# MMA3001 - Temperature Reconstruction
# Stage 1: Exploring raw environmental sensor data

from pathlib import Path
import zipfile
import pandas as pd
import json

# Locate the zipped dataset
data_path = Path("data/5EnvSensor_MayToDec2024_180kRows.zip")

# Inspect the first 1000 raw records
with zipfile.ZipFile(data_path) as archive:
    filename = archive.namelist()[0]

    with archive.open(filename) as file:
        df = pd.read_csv(file, nrows=1000)

print("Dataset columns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nNumber of different sensors in sample:")
print(df["sensorid"].nunique())

# Examine the environmental measurements
sample = json.loads(df.loc[0, "jsondata"])

print("\nMeasurements available:")
for measurement in sample:
    variable = measurement["variable"]
    print(variable["name"], "-", variable["unit"])
