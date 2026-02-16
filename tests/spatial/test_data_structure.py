"""Check data structure for SAR RE."""

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_PATH = Path(__file__).parent / "fixtures"

# Load data
df = pd.read_csv(FIXTURES_PATH / "spatial_test_data.csv")

print("First 30 rows (entity, time):")
print(df[["entity", "time"]].head(30))

print("\n\nLast 30 rows (entity, time):")
print(df[["entity", "time"]].tail(30))

print("\n\nData order check:")
print(f"Is data sorted by (entity, time)? {df.equals(df.sort_values(['entity', 'time']))}")
print(f"Is data sorted by (time, entity)? {df.equals(df.sort_values(['time', 'entity']))}")

# Check if data is entity-major (entity 1 all times, entity 2 all times, ...)
entity_time_pairs = list(zip(df["entity"], df["time"]))
is_entity_major = all(
    entity_time_pairs[i][0] <= entity_time_pairs[i + 1][0]
    for i in range(len(entity_time_pairs) - 1)
)
print(f"Is data entity-major order? {is_entity_major}")

# Check expected order for entity-major
expected_entity_major = []
for entity in range(1, 31):
    for time in range(1, 11):
        expected_entity_major.append((entity, time))

matches_expected = entity_time_pairs == expected_entity_major
print(f"Matches expected entity-major order? {matches_expected}")

# Check current order
print("\n\nCurrent order pattern (first 30 rows):")
for i in range(min(30, len(df))):
    entity, time = df.iloc[i]["entity"], df.iloc[i]["time"]
    print(f"  Row {i}: entity={entity}, time={time}")
