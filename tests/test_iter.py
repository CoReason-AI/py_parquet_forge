# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forget

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from py_parquet_forge.main import read_parquet_iter, write_parquet, write_to_dataset


@pytest.fixture
def sample_parquet_file(tmp_path):
    """Creates a sample Parquet file for testing."""
    file_path = tmp_path / "test.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("value", pa.string()),
        ]
    )
    data = [{"id": i, "value": f"value_{i}"} for i in range(100)]
    write_parquet(data, file_path, schema)
    return file_path


@pytest.fixture
def sample_partitioned_dataset(tmp_path):
    """Creates a sample partitioned Parquet dataset."""
    dataset_path = tmp_path / "dataset"
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("partition_key", pa.string()),
        ]
    )
    data = [{"id": i, "partition_key": "A" if i < 50 else "B"} for i in range(100)]
    write_to_dataset(data, dataset_path, schema, partition_cols=["partition_key"])
    return dataset_path


def test_read_parquet_iter_single_file(sample_parquet_file):
    """Verify iterating over a single Parquet file yields all data."""
    # Act
    batches = list(read_parquet_iter(sample_parquet_file, chunk_size=30))

    # Assert
    assert len(batches) == 4  # 100 rows / 30 chunk_size = 3 full batches + 1 partial
    assert sum(batch.num_rows for batch in batches) == 100
    assert batches[0].num_rows == 30
    assert batches[1].num_rows == 30
    assert batches[2].num_rows == 30
    assert batches[3].num_rows == 10


def test_read_parquet_iter_dataset(sample_partitioned_dataset):
    """Verify iterating over a partitioned dataset yields all data."""
    # Act
    batches = list(read_parquet_iter(sample_partitioned_dataset, chunk_size=40))

    # Assert
    # pyarrow scans file by file. 50 rows in file A -> 2 batches (40, 10)
    # 50 rows in file B -> 2 batches (40, 10). Total = 4 batches.
    assert len(batches) == 4
    assert sum(batch.num_rows for batch in batches) == 100
    # Check that the partition column is included
    assert "partition_key" in batches[0].schema.names


def test_read_parquet_iter_with_columns(sample_parquet_file):
    """Verify that column projection works correctly."""
    # Act
    batches = list(read_parquet_iter(sample_parquet_file, columns=["id"]))

    # Assert
    assert sum(batch.num_rows for batch in batches) == 100
    assert batches[0].schema.names == ["id"]


def test_read_parquet_iter_with_filters(sample_parquet_file):
    """Verify that predicate pushdown (filters) works correctly."""
    # Act
    # Filter for rows where id >= 95
    filters = ds.field("id") >= 95
    batches = list(read_parquet_iter(sample_parquet_file, filters=filters))

    # Assert
    assert sum(batch.num_rows for batch in batches) == 5
    first_batch_ids = batches[0].column("id").to_pylist()
    assert first_batch_ids == [95, 96, 97, 98, 99]


def test_read_parquet_iter_empty_file(tmp_path):
    """Verify iterating over an empty file yields no data."""
    # Arrange
    file_path = tmp_path / "empty.parquet"
    schema = pa.schema([pa.field("id", pa.int64())])
    write_parquet([], file_path, schema)

    # Act
    batches = list(read_parquet_iter(file_path))

    # Assert
    # The total number of rows should be 0, even if a schema-only batch is returned.
    assert sum(batch.num_rows for batch in batches) == 0


def test_read_parquet_iter_smaller_than_chunk_size(tmp_path):
    """Verify iterating a file with fewer rows than chunk_size yields one batch."""
    # Arrange
    file_path = tmp_path / "small.parquet"
    schema = pa.schema([pa.field("id", pa.int64())])
    data = [{"id": i} for i in range(10)]
    write_parquet(data, file_path, schema)

    # Act
    batches = list(read_parquet_iter(file_path, chunk_size=100))

    # Assert
    assert len(batches) == 1
    assert batches[0].num_rows == 10
