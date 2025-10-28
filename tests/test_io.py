# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from py_parquet_forge.main import inspect_schema


def test_inspect_schema_single_file(tmp_path):
    """Verify that inspect_schema correctly reads the schema from a single Parquet file."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    expected_schema = pa.schema([
        pa.field("col1", pa.int64()),
        pa.field("col2", pa.string()),
    ])
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
    })
    table = pa.Table.from_pandas(df, schema=expected_schema)
    pq.write_table(table, output_path)

    # Act
    actual_schema = inspect_schema(output_path)

    # Assert
    assert actual_schema.equals(expected_schema)


def test_inspect_schema_nonexistent_path(tmp_path):
    """Verify that inspect_schema raises an exception for a nonexistent path."""
    # Arrange
    nonexistent_path = tmp_path / "nonexistent"

    # Act & Assert
    with pytest.raises(pa.ArrowIOError):
        inspect_schema(nonexistent_path)


def test_inspect_schema_invalid_file(tmp_path):
    """Verify that inspect_schema raises an exception for an invalid file type."""
    # Arrange
    invalid_file = tmp_path / "invalid.txt"
    with open(invalid_file, "w") as f:
        f.write("this is not a parquet file")

    # Act & Assert
    with pytest.raises(pa.ArrowInvalid):
        inspect_schema(invalid_file)


def test_inspect_schema_dataset_directory(tmp_path):
    """Verify that inspect_schema correctly reads the schema from a dataset directory."""
    # Arrange
    output_dir = tmp_path / "dataset"
    write_schema = pa.schema([
        pa.field("col1", pa.int64()),
        pa.field("col2", pa.string()),
        pa.field("partition_col", pa.string()),
    ])
    data = {
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
        "partition_col": ["one", "two", "one"]
    }
    table = pa.Table.from_pydict(data, schema=write_schema)
    pq.write_to_dataset(table, root_path=output_dir, partition_cols=["partition_col"])

    # Act
    actual_schema = inspect_schema(output_dir)

    # When a dataset is read, the partition column is dictionary-encoded by default.
    # We must construct the expected schema to reflect this for a valid comparison.
    expected_schema_after_read = pa.schema([
        pa.field("col1", pa.int64()),
        pa.field("col2", pa.string()),
        pa.field("partition_col", pa.dictionary(pa.int32(), pa.string(), ordered=False))
    ])

    # Assert
    assert actual_schema.equals(expected_schema_after_read)
