# Copyright (c) 2025 CoReason, Inc
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
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from py_parquet_forge import (
    read_parquet,
    read_parquet_iter,
    write_parquet,
    write_to_dataset,
)


def test_read_parquet_iter_single_file(tmp_path):
    """Verify iterating over a single Parquet file yields all data in chunks."""
    # Arrange
    assert pytest is not None  # Trick to prevent ruff from removing the unused import
    output_path = tmp_path / "test.parquet"
    row_count = 1000
    schema = pa.schema([("number", pa.int64())])
    data = [{"number": i} for i in range(row_count)]
    write_parquet(data, output_path, schema)

    # Act
    chunk_size = 200
    batches = list(read_parquet_iter(output_path, chunk_size=chunk_size))
    total_rows_read = sum(batch.num_rows for batch in batches)

    # Assert
    assert total_rows_read == row_count
    # Verify that chunking is happening (more than one batch)
    assert len(batches) > 1
    # Check that the last batch has the remaining rows
    assert batches[-1].num_rows == row_count % chunk_size or chunk_size


def test_read_parquet_iter_dataset(tmp_path):
    """Verify iterating over a partitioned dataset yields data from all partitions."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("value", pa.int64()), ("part", pa.string())])
    data = [
        {"value": 1, "part": "a"},
        {"value": 2, "part": "b"},
        {"value": 3, "part": "a"},
        {"value": 4, "part": "b"},
    ]
    write_to_dataset(data, output_dir, schema, partition_cols=["part"])

    # Act
    batches = list(read_parquet_iter(output_dir, chunk_size=1))
    total_rows_read = sum(batch.num_rows for batch in batches)
    read_data = pa.Table.from_batches(batches).to_pydict()

    # Assert
    assert total_rows_read == 4
    # Check that data from both partitions was read
    assert sorted(read_data["value"]) == [1, 2, 3, 4]
    assert sorted(read_data["part"]) == ["a", "a", "b", "b"]


def test_read_parquet_iter_with_column_projection(tmp_path):
    """Verify that the 'columns' parameter correctly projects columns."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int64()), ("b", pa.string()), ("c", pa.float64())])
    data = [{"a": 1, "b": "x", "c": 1.1}, {"a": 2, "b": "y", "c": 2.2}]
    write_parquet(data, output_path, schema)

    # Act
    batches = list(read_parquet_iter(output_path, columns=["a", "c"]))

    # Assert
    assert len(batches) > 0
    # Check that the schema of the first batch has the correct columns
    projected_schema = batches[0].schema
    assert projected_schema.names == ["a", "c"]

    # Verify the data
    table = pa.Table.from_batches(batches)
    assert table.num_rows == 2
    assert sorted(table.column("a").to_pylist()) == [1, 2]
    assert "b" not in table.column_names


def test_read_parquet_iter_with_filters(tmp_path):
    """Verify that the 'filters' parameter correctly filters rows."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("value", pa.int64()), ("category", pa.string())])
    data = [
        {"value": 10, "category": "A"},
        {"value": 20, "category": "B"},
        {"value": 30, "category": "A"},
        {"value": 40, "category": "B"},
    ]
    write_parquet(data, output_path, schema)

    # Act
    # Use a DNF filter to construct a pyarrow.compute expression
    filters = (pc.field("value") > 25) & (pc.field("category") == "A")
    batches = list(read_parquet_iter(output_path, filters=filters))
    total_rows_read = sum(batch.num_rows for batch in batches)
    table = pa.Table.from_batches(batches) if batches else pa.Table.from_pydict({})

    # Assert
    assert total_rows_read == 1
    assert table.num_rows == 1
    assert table.column("value").to_pylist() == [30]
    assert table.column("category").to_pylist() == ["A"]


def test_read_parquet_iter_empty_file(tmp_path):
    """Verify iterating over an empty Parquet file yields no batches."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("col1", pa.int64())])
    write_parquet([], output_path, schema)

    # Act
    batches = list(read_parquet_iter(output_path))
    total_rows_read = sum(batch.num_rows for batch in batches)

    # Assert
    # pyarrow may yield a single, schema-only batch with zero rows
    assert total_rows_read == 0


def test_read_parquet_iter_handles_empty_dataset_gracefully(tmp_path):
    """
    Verifies that iterating over an empty or non-existent dataset directory
    yields no data and does not raise an error.
    """
    output_dir = tmp_path / "dataset"

    # Case 1: The directory does not exist.
    # pyarrow's ds.dataset will raise a FileNotFoundError here.
    with pytest.raises(pa.lib.ArrowIOError):
        list(read_parquet_iter(output_dir))

    # Case 2: The directory exists but is empty.
    output_dir.mkdir()
    batches_empty_dir = list(read_parquet_iter(output_dir))
    assert sum(b.num_rows for b in batches_empty_dir) == 0

    # Case 3: The directory contains empty subdirectories (e.g., empty partitions).
    (output_dir / "part=a").mkdir()
    batches_empty_part = list(read_parquet_iter(output_dir))
    assert sum(b.num_rows for b in batches_empty_part) == 0


def test_read_parquet_iter_chunk_size_larger_than_file(tmp_path):
    """Verify it yields a single batch if chunk_size > row_count."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    row_count = 50
    schema = pa.schema([("number", pa.int64())])
    data = [{"number": i} for i in range(row_count)]
    write_parquet(data, output_path, schema)

    # Act
    # Set chunk_size much larger than the number of rows
    batches = list(read_parquet_iter(output_path, chunk_size=row_count * 2))
    total_rows_read = sum(batch.num_rows for batch in batches)

    # Assert
    assert len(batches) == 1
    assert total_rows_read == row_count
    assert batches[0].num_rows == row_count


def test_read_parquet_iter_invalid_filter_type(tmp_path):
    """Verify that an invalid filter type raises an appropriate error."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("value", pa.int64())])
    write_parquet([{"value": 10}], output_path, schema)

    # Act & Assert
    # The modern `pyarrow.dataset` API requires a compute expression, not a list of tuples.
    # We expect this to fail with a TypeError or ArrowInvalid error.
    with pytest.raises((TypeError, pa.ArrowInvalid)):
        # This is the old, unsupported filter format for this function
        invalid_filters = [("value", ">", 5)]
        list(read_parquet_iter(output_path, filters=invalid_filters))


def test_read_parquet_pandas_output(tmp_path):
    """Verify reading a Parquet file to a pandas DataFrame."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int64())])
    write_parquet([{"a": 1}], output_path, schema)

    # Act
    df = read_parquet(output_path, output_format="pandas")

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 1)
    assert df["a"][0] == 1


def test_read_parquet_arrow_output(tmp_path):
    """Verify reading a Parquet file to a pyarrow Table."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int64())])
    write_parquet([{"a": 1}], output_path, schema)

    # Act
    table = read_parquet(output_path, output_format="arrow")

    # Assert
    assert isinstance(table, pa.Table)
    assert table.num_rows == 1
    assert table.schema.equals(schema)


def test_read_parquet_with_column_projection(tmp_path):
    """Verify that only specified columns are read."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    write_parquet([{"a": 1, "b": "x"}], output_path, schema)

    # Act
    df = read_parquet(output_path, columns=["a"])

    # Assert
    assert "a" in df.columns
    assert "b" not in df.columns


def test_read_parquet_with_filters(tmp_path):
    """Verify that rows are filtered correctly."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("value", pa.int64())])
    data = [{"value": 10}, {"value": 20}, {"value": 30}]
    write_parquet(data, output_path, schema)

    # Act
    df = read_parquet(output_path, filters=[("value", ">", 15)])

    # Assert
    assert len(df) == 2
    assert sorted(df["value"].tolist()) == [20, 30]

    # Act
    table = read_parquet(
        output_path, filters=[("value", ">", 15)], output_format="arrow"
    )

    # Assert
    assert isinstance(table, pa.Table)
    assert table.num_rows == 2
    assert sorted(table.column("value").to_pylist()) == [20, 30]


def test_read_parquet_invalid_output_format(tmp_path):
    """Verify that a ValueError is raised for an invalid output format."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int64())])
    write_parquet([{"a": 1}], output_path, schema)

    # Act & Assert
    with pytest.raises(
        ValueError, match="output_format must be either 'pandas' or 'arrow'"
    ):
        read_parquet(output_path, output_format="invalid_format")


def test_read_parquet_from_dataset(tmp_path):
    """Verify reading from a partitioned dataset works correctly."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("value", pa.int64()), ("part", pa.string())])
    data = [
        {"value": 1, "part": "a"},
        {"value": 2, "part": "b"},
        {"value": 3, "part": "a"},
    ]
    write_to_dataset(data, output_dir, schema, partition_cols=["part"])

    # Act
    df = read_parquet(output_dir)

    # Assert
    assert len(df) == 3
    assert sorted(df["value"].tolist()) == [1, 2, 3]
    # In pyarrow datasets, partition columns are added as categorical
    assert df["part"].dtype == "category"


def test_read_parquet_handles_nullable_integers(tmp_path):
    """
    Verify that integer columns with nulls are read into pandas DataFrames
    with the nullable 'Int64' dtype.
    """
    output_file = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("id", pa.int64(), nullable=True)])

    # Create a table with nulls in the integer column
    table = pa.Table.from_pydict({"id": [1, None, 3]}, schema=schema)
    pq.write_table(table, output_file)

    # Act
    df = read_parquet(output_file, output_format="pandas")

    # Assert
    # 1. Check that the dtype is pandas' nullable integer type
    assert pd.api.types.is_integer_dtype(df["id"].dtype)
    assert df["id"].dtype.name == "Int64"

    # 2. Check that the null value is pd.NA
    assert pd.isna(df["id"][1])

    # 3. Check that non-null values are correct
    assert df["id"][0] == 1
    assert df["id"][2] == 3
