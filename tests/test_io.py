# Copyright (c) 2025 CoReason, Inc.
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
from py_parquet_forge.main import inspect_schema, write_parquet, write_to_dataset


def test_write_parquet_success_pandas(tmp_path):
    """Verify writing a pandas DataFrame to a Parquet file succeeds."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
        ]
    )
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }
    )

    # Act
    write_parquet(df, output_path, schema)

    # Assert
    assert output_path.exists()
    written_schema = pq.read_schema(output_path)
    assert written_schema.equals(schema)
    read_table = pq.read_table(output_path)
    assert read_table.num_rows == 3


def test_write_parquet_success_pydict(tmp_path):
    """Verify writing a list of dictionaries to a Parquet file succeeds."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("value", pa.float64()),
        ]
    )
    data = [
        {"id": 1, "value": 1.1},
        {"id": 2, "value": 2.2},
    ]

    # Act
    write_parquet(data, output_path, schema)

    # Assert
    assert output_path.exists()
    written_schema = pq.read_schema(output_path)
    assert written_schema.equals(schema)


def test_write_parquet_schema_validation_error(tmp_path):
    """Verify SchemaValidationError is raised for incompatible data."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    # Data has a string where an int is expected
    df = pd.DataFrame({"a": [1, 2, "not-an-int"]})

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        write_parquet(df, output_path, schema)

    # Assert that no file was created
    assert not output_path.exists()


def test_write_parquet_atomicity_on_failure(tmp_path):
    """Verify that no partial file is left if writing fails mid-way."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("col1", pa.int64())])
    df = pd.DataFrame({"col1": [1, 2, 3]})

    # Create a pre-existing file to ensure it's not touched on failure
    pre_existing_content = "pre-existing"
    output_path.write_text(pre_existing_content)

    # Mock pq.write_table to raise an exception during the write operation
    with patch("pyarrow.parquet.write_table", side_effect=IOError("Disk full!")):
        # Act & Assert
        with pytest.raises(IOError):
            write_parquet(df, output_path, schema)

    # Assert that the original file is untouched and no temp file exists
    assert output_path.read_text() == pre_existing_content

    # Check that no .tmp files are left in the directory
    temp_files = list(tmp_path.glob("*.tmp"))
    assert not temp_files, f"Temp files found: {temp_files}"


def test_write_parquet_os_error_on_cleanup(tmp_path):
    """Verify that an OSError during cleanup is logged but not propagated."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": [1]})

    # Mock os.replace to allow the temporary file to be created but not renamed
    with patch("os.replace"):
        # Mock os.remove to raise an OSError
        with patch("os.remove", side_effect=OSError("Permission denied")):
            # Act
            write_parquet(df, output_path, schema)

    # Assert
    # The test passes if no exception is raised


def test_write_parquet_exception_on_replace(tmp_path):
    """Verify that an exception during replace is handled correctly."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": [1]})

    # Mock os.replace to raise an exception
    with patch("os.replace", side_effect=Exception("Test exception")):
        # Act & Assert
        with pytest.raises(Exception):
            write_parquet(df, output_path, schema)


def test_write_parquet_overwrites_existing_file(tmp_path):
    """Verify that write_parquet overwrites an existing file."""
    # Arrange
    output_path = tmp_path / "test.parquet"

    # First write
    schema1 = pa.schema([pa.field("a", pa.int32())])
    df1 = pd.DataFrame({"a": [1]})
    write_parquet(df1, output_path, schema1)

    table1 = pq.read_table(output_path)
    assert table1.num_rows == 1
    assert table1.schema.equals(schema1)

    # Second write (overwrite)
    schema2 = pa.schema([pa.field("b", pa.string())])
    df2 = pd.DataFrame({"b": ["x", "y"]})

    # Act
    write_parquet(df2, output_path, schema2)

    # Assert
    assert output_path.exists()
    table2 = pq.read_table(output_path)
    assert table2.num_rows == 2
    assert table2.schema.equals(schema2)


def test_inspect_schema_single_file(tmp_path):
    """Verify that inspect_schema correctly reads the schema from a single Parquet file."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    expected_schema = pa.schema(
        [
            pa.field("col1", pa.int64()),
            pa.field("col2", pa.string()),
        ]
    )
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        }
    )
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


def test_write_parquet_empty_pydict(tmp_path):
    """Verify that an empty list of dicts can be written to a Parquet file."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    data = []

    # Act
    write_parquet(data, output_path, schema)

    # Assert
    assert output_path.exists()
    written_schema = pq.read_schema(output_path)
    assert written_schema.equals(schema)
    read_table = pq.read_table(output_path)
    assert read_table.num_rows == 0


def test_write_parquet_path_with_spaces(tmp_path):
    """Verify writing to a path with spaces succeeds."""
    # Arrange
    output_path = tmp_path / "path with spaces" / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    data = [{"a": 1}]

    # Act
    write_parquet(data, output_path, schema)

    # Assert
    assert output_path.exists()
    read_table = pq.read_table(output_path)
    assert read_table.num_rows == 1


def test_inspect_schema_dataset_directory(tmp_path):
    """Verify that inspect_schema correctly reads the schema from a dataset directory."""
    # Arrange
    output_dir = tmp_path / "dataset"
    write_schema = pa.schema(
        [
            pa.field("col1", pa.int64()),
            pa.field("col2", pa.string()),
            pa.field("partition_col", pa.string()),
        ]
    )
    data = {
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
        "partition_col": ["one", "two", "one"],
    }
    table = pa.Table.from_pydict(data, schema=write_schema)
    pq.write_to_dataset(table, root_path=output_dir, partition_cols=["partition_col"])

    # Act
    actual_schema = inspect_schema(output_dir)

    # When a dataset is read, the partition column is dictionary-encoded by default.
    # We must construct the expected schema to reflect this for a valid comparison.
    expected_schema_after_read = pa.schema(
        [
            pa.field("col1", pa.int64()),
            pa.field("col2", pa.string()),
            pa.field(
                "partition_col", pa.dictionary(pa.int32(), pa.string(), ordered=False)
            ),
        ]
    )

    # Assert
    assert actual_schema.equals(expected_schema_after_read)


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
