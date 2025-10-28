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
from py_parquet_forge.main import inspect_schema, write_parquet


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

    # Mock os.rename to allow the temporary file to be created but not renamed
    with patch("os.rename"):
        # Mock os.remove to raise an OSError
        with patch("os.remove", side_effect=OSError("Permission denied")):
            # Act
            write_parquet(df, output_path, schema)

    # Assert
    # The test passes if no exception is raised


def test_write_parquet_exception_on_rename(tmp_path):
    """Verify that an exception during rename is handled correctly."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": [1]})

    # Mock os.rename to raise an exception
    with patch("os.rename", side_effect=Exception("Test exception")):
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
