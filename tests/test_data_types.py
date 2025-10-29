# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from py_parquet_forge import read_parquet, write_parquet


def test_data_type_timestamp_nanos(tmp_path: Path):
    """Verify that high-precision timestamps (nanoseconds) are preserved."""
    # Arrange
    assert pytest  # Trick to prevent ruff from removing the unused import
    schema = pa.schema(
        [
            pa.field("ts_ns", pa.timestamp("ns")),
        ]
    )
    # Use pandas to create high-precision timestamps, which are natively nanosecond
    ts = pd.Timestamp("2023-01-01 12:34:56.789123456")
    df = pd.DataFrame({"ts_ns": [ts]})
    output_path = tmp_path / "test.parquet"

    # Act
    write_parquet(df, output_path, schema)
    read_df = read_parquet(output_path)

    # Assert
    assert read_df["ts_ns"][0] == ts


def test_data_type_timestamp_with_timezone(tmp_path: Path):
    """Verify that timestamps with timezone information are preserved."""
    # Arrange
    schema = pa.schema(
        [
            pa.field("ts_tz", pa.timestamp("us", tz="America/New_York")),
        ]
    )
    ts = pd.Timestamp("2023-01-01 12:00:00", tz="America/New_York")
    df = pd.DataFrame({"ts_tz": [ts]})
    output_path = tmp_path / "test.parquet"

    # Act
    write_parquet(df, output_path, schema)
    read_df = read_parquet(output_path)

    # Assert
    assert read_df["ts_tz"][0] == ts
    assert str(read_df["ts_tz"][0].tz) == "America/New_York"


def test_data_type_decimal(tmp_path: Path):
    """Verify that Decimal128 types are preserved."""
    # Arrange
    schema = pa.schema(
        [
            pa.field("price", pa.decimal128(10, 2)),
        ]
    )
    data = [{"price": Decimal("199.99")}, {"price": Decimal("-0.50")}, {"price": None}]
    output_path = tmp_path / "test.parquet"

    # Act
    write_parquet(data, output_path, schema)
    read_df = read_parquet(output_path)

    # Assert
    # PyArrow reads decimals as Python Decimal objects
    assert read_df["price"].tolist() == [Decimal("199.99"), Decimal("-0.50"), None]


def test_data_type_nested_list(tmp_path: Path):
    """Verify that nested lists are correctly handled."""
    # Arrange
    schema = pa.schema([pa.field("scores", pa.list_(pa.int32()))])
    data = [
        {"scores": [10, 20, 30]},
        {"scores": None},
        {"scores": []},
    ]
    output_path = tmp_path / "test.parquet"

    # Act
    write_parquet(data, output_path, schema)
    read_df = read_parquet(output_path, output_format="arrow")

    # Assert
    # Using an Arrow table for comparison is more robust for list types
    expected_data = [[10, 20, 30], None, []]
    assert read_df.column("scores").to_pylist() == expected_data


def test_data_type_nested_struct(tmp_path: Path):
    """Verify that nested structs are correctly handled."""
    # Arrange
    struct_type = pa.struct([pa.field("x", pa.int64()), pa.field("y", pa.string())])
    schema = pa.schema([pa.field("point", struct_type)])
    data = [
        {"point": {"x": 1, "y": "a"}},
        {"point": None},
        {"point": {"x": 2, "y": "b"}},
    ]
    output_path = tmp_path / "test.parquet"

    # Act
    write_parquet(data, output_path, schema)
    read_table = read_parquet(output_path, output_format="arrow")

    # Assert
    expected_data = [
        {"x": 1, "y": "a"},
        None,
        {"x": 2, "y": "b"},
    ]
    assert read_table.column("point").to_pylist() == expected_data


def test_data_type_binary(tmp_path: Path):
    """Verify that binary data is correctly preserved."""
    # Arrange
    schema = pa.schema([pa.field("data", pa.binary())])
    binary_data = [b"\x01\x02\x03", b"hello", None]
    data = [{"data": d} for d in binary_data]
    output_path = tmp_path / "test.parquet"

    # Act
    write_parquet(data, output_path, schema)
    read_table = read_parquet(output_path, output_format="arrow")

    # Assert
    assert read_table.column("data").to_pylist() == binary_data
