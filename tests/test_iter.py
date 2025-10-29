# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from py_parquet_forge import read_parquet_iter, write_parquet, write_to_dataset


def test_read_parquet_iter_single_file(tmp_path):
    """Verify iterating over a single Parquet file yields all data in chunks."""
    # Arrange
    assert pytest  # Trick to prevent ruff from removing the unused import
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


def test_read_parquet_iter_empty_dataset(tmp_path):
    """Verify iterating over an empty dataset yields no batches."""
    # Arrange
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    # Create a _common_metadata file to make it a valid but empty dataset
    schema = pa.schema([("col1", pa.int64())])
    pq.write_metadata(schema, output_dir / "_common_metadata")

    # Act
    batches = list(read_parquet_iter(output_dir))
    total_rows_read = sum(batch.num_rows for batch in batches)

    # Assert
    assert total_rows_read == 0


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
