# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from py_parquet_forge.exceptions import SchemaValidationError
from py_parquet_forge.main import write_to_dataset


def test_write_to_dataset_append_mode(tmp_path):
    """Verify writing in append mode adds new data without removing existing data."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    df1 = pd.DataFrame({"col1": [1, 2]})
    df2 = pd.DataFrame({"col1": [3, 4]})

    # Act
    write_to_dataset(df1, output_dir, schema, mode="append")
    write_to_dataset(df2, output_dir, schema, mode="append")

    # Assert
    dataset = pq.ParquetDataset(output_dir)
    table = dataset.read()
    assert table.num_rows == 4
    assert sorted(table.column("col1").to_pylist()) == [1, 2, 3, 4]


def test_write_to_dataset_overwrite_mode(tmp_path):
    """Verify writing in overwrite mode removes existing data before writing new data."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    df1 = pd.DataFrame({"col1": [1, 2]})
    df2 = pd.DataFrame({"col1": [3, 4]})

    # Act
    write_to_dataset(df1, output_dir, schema, mode="append")  # Initial write
    write_to_dataset(df2, output_dir, schema, mode="overwrite")  # Overwrite

    # Assert
    dataset = pq.ParquetDataset(output_dir)
    table = dataset.read()
    assert table.num_rows == 2
    assert sorted(table.column("col1").to_pylist()) == [3, 4]


def test_write_to_dataset_with_partitioning(tmp_path):
    """Verify that data is correctly partitioned into subdirectories."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("value", pa.int64()), ("part", pa.string())])
    df = pd.DataFrame({"value": [1, 2, 3], "part": ["a", "b", "a"]})

    # Act
    write_to_dataset(df, output_dir, schema, partition_cols=["part"])

    # Assert
    part_a_path = output_dir / "part=a"
    part_b_path = output_dir / "part=b"
    assert part_a_path.is_dir()
    assert part_b_path.is_dir()

    # Verify content of partition 'a'
    table_a = pq.read_table(part_a_path)
    assert table_a.num_rows == 2
    assert sorted(table_a.column("value").to_pylist()) == [1, 3]

    # Verify content of partition 'b'
    table_b = pq.read_table(part_b_path)
    assert table_b.num_rows == 1
    assert table_b.column("value").to_pylist() == [2]


def test_write_to_dataset_schema_validation_error(tmp_path):
    """Verify that SchemaValidationError is raised for invalid data."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    df = pd.DataFrame({"col1": ["not-an-int"]})

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        write_to_dataset(df, output_dir, schema)
    assert not output_dir.exists()


def test_write_to_dataset_invalid_mode(tmp_path):
    """Verify that a ValueError is raised for an invalid mode."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    df = pd.DataFrame({"col1": [1]})

    # Act & Assert
    with pytest.raises(ValueError, match="mode must be either 'append' or 'overwrite'"):
        write_to_dataset(df, output_dir, schema, mode="invalid_mode")


def test_write_to_dataset_pydict_input(tmp_path):
    """Verify that a list of dictionaries can be written to a dataset."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("id", pa.int64())])
    data = [{"id": 1}, {"id": 2}]

    # Act
    write_to_dataset(data, output_dir, schema)

    # Assert
    table = pq.read_table(output_dir)
    assert table.num_rows == 2
    assert table.schema.equals(pa.schema([pa.field("id", pa.int64())]))


def test_write_to_dataset_empty_input(tmp_path):
    """Verify writing empty data creates the directory but no files."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    empty_df = pd.DataFrame({"col1": []})

    # Act
    write_to_dataset(empty_df, output_dir, schema)

    # Assert
    assert output_dir.is_dir()
    # Pyarrow does not write any files (including metadata) for an empty table
    assert not any(output_dir.iterdir())


def test_write_to_dataset_overwrite_os_error(tmp_path):
    """Verify that an OSError during directory removal is propagated."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    df = pd.DataFrame({"col1": [1]})
    write_to_dataset(df, output_dir, schema)  # Create the directory

    # Act & Assert
    with patch("shutil.rmtree", side_effect=OSError("Permission denied")):
        with pytest.raises(OSError, match="Permission denied"):
            write_to_dataset(df, output_dir, schema, mode="overwrite")


def test_write_to_dataset_invalid_partition_column(tmp_path):
    """Verify that an error is raised when a partition column does not exist."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("a", pa.int32())])
    data = [{"a": 1}]

    # Act & Assert
    with pytest.raises(
        pa.ArrowInvalid, match="Partition column 'non_existent_col' not in schema"
    ):
        write_to_dataset(data, output_dir, schema, partition_cols=["non_existent_col"])


def test_write_to_dataset_with_empty_partition_cols(tmp_path):
    """Verify that partition_cols=[] is treated as no partitioning."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("value", pa.int64()), ("part", pa.string())])
    df = pd.DataFrame({"value": [1, 2, 3], "part": ["a", "b", "a"]})

    # Act
    write_to_dataset(df, output_dir, schema, partition_cols=[])

    # Assert
    # Check that no partition subdirectories were created
    assert not any(f.is_dir() for f in output_dir.iterdir())

    # Check that at least one parquet file was created in the root
    parquet_files = list(output_dir.glob("*.parquet"))
    assert len(parquet_files) >= 1

    # Verify the content of the dataset
    table = pq.read_table(output_dir)
    assert table.num_rows == 3
    assert "part" in table.schema.names
    assert "value" in table.schema.names


def test_write_to_dataset_mkdir_os_error(tmp_path):
    """Verify that an OSError during directory creation is propagated."""
    # Arrange
    output_dir = tmp_path / "nonexistent"
    schema = pa.schema([("a", pa.int32())])
    data = [{"a": 1}]

    # Act & Assert
    with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
        with pytest.raises(OSError, match="Permission denied"):
            write_to_dataset(data, output_dir, schema)


def test_overwrite_mode_does_not_delete_on_validation_error(tmp_path):
    """
    Verifies that in 'overwrite' mode, the existing dataset is NOT deleted
    if the new data fails schema validation.
    """
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])

    # 1. Create an initial, valid dataset.
    initial_df = pd.DataFrame({"col1": [1, 2]})
    write_to_dataset(initial_df, output_dir, schema, mode="append")
    initial_table = pq.read_table(output_dir)
    assert initial_table.num_rows == 2

    # 2. Attempt to overwrite with invalid data.
    invalid_df = pd.DataFrame({"col1": ["not-an-int"]})
    with pytest.raises(SchemaValidationError):
        write_to_dataset(invalid_df, output_dir, schema, mode="overwrite")

    # 3. Assert that the original dataset still exists and is unchanged.
    assert output_dir.exists()
    final_table = pq.read_table(output_dir)
    assert final_table.equals(initial_table)


def test_write_to_dataset_logs_arrow_exception(tmp_path, mocker):
    """
    Verifies that an ArrowException during the write operation is logged.
    """
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("a", pa.int32())])
    data = [{"a": 1}]
    mock_logger_error = mocker.patch("py_parquet_forge.main.logger.error")

    with patch("pyarrow.parquet.write_to_dataset", side_effect=pa.ArrowException("Test error")):
        with pytest.raises(SchemaValidationError):
            write_to_dataset(data, output_dir, schema)

    mock_logger_error.assert_called_once()
    assert "Arrow schema validation error" in mock_logger_error.call_args[0][0]
    assert "Test error" in mock_logger_error.call_args[0][0]
