# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from py_parquet_forge import write_parquet
from py_parquet_forge.exceptions import SchemaValidationError

# region Test Data and Schema Definitions

SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("name", pa.string()),
        pa.field("value", pa.float64()),
    ]
)

PANDAS_DF = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "value": [10.1, 20.2, 30.3],
    }
)

LIST_OF_DICTS = PANDAS_DF.to_dict("records")
ARROW_TABLE = pa.Table.from_pandas(PANDAS_DF, schema=SCHEMA, preserve_index=False)
ARROW_BATCH = pa.RecordBatch.from_pandas(PANDAS_DF, schema=SCHEMA, preserve_index=False)

# endregion

# region Positive Test Cases


@pytest.mark.parametrize(
    "input_data",
    [
        PANDAS_DF,
        LIST_OF_DICTS,
        ARROW_TABLE,
        ARROW_BATCH,
    ],
    ids=["pandas_df", "list_of_dicts", "arrow_table", "arrow_batch"],
)
def test_write_parquet_all_input_types(tmp_path: Path, input_data):
    """Verifies that write_parquet can handle all supported InputData types."""
    output_file = tmp_path / "test.parquet"
    write_parquet(input_data, output_file, SCHEMA)

    assert output_file.exists()
    table = pa.parquet.read_table(output_file)
    assert table.schema.equals(SCHEMA, check_metadata=False)
    assert table.num_rows == 3
    pd.testing.assert_frame_equal(PANDAS_DF, table.to_pandas())


def test_write_parquet_kwargs_passthrough(tmp_path: Path):
    """
    Verifies that **kwargs are correctly passed through to the underlying
    pyarrow.parquet.write_table function.
    """
    output_path = tmp_path / "test.parquet"

    # Use a kwarg that has a verifiable side-effect, like write_statistics.
    write_parquet(ARROW_TABLE, output_path, SCHEMA, write_statistics=False)

    # Inspect the written file's metadata.
    parquet_file = pq.ParquetFile(output_path)
    metadata = parquet_file.metadata

    # Check that statistics are not written for the column chunk.
    column_chunk = metadata.row_group(0).column(0)
    assert column_chunk.statistics is None


# endregion


def test_write_parquet_schema_validation_error_incompatible_data(tmp_path):
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
    """Verify that an exception during replace is handled correctly and the temp file is cleaned up."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": [1]})

    # Mock os.replace to raise an exception to trigger the finally block
    with patch("os.replace", side_effect=IOError("Move failed")):
        # Act & Assert
        with pytest.raises(IOError):
            write_parquet(df, output_path, schema)

    # Assert that the temporary file was successfully cleaned up
    temp_files = list(tmp_path.glob("*.tmp"))
    assert not temp_files, f"Temp files were not cleaned up: {temp_files}"


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


def test_write_parquet_with_na_values(tmp_path):
    """Verify that pd.NA values are correctly handled."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32()), pa.field("b", pa.float64())])
    df = pd.DataFrame({"a": [1, pd.NA, 3], "b": [4.0, 5.0, pd.NA]})

    # Act
    write_parquet(df, output_path, schema)

    # Assert
    assert output_path.exists()
    table = pq.read_table(output_path)
    assert table.schema.equals(schema)
    assert table.column("a").to_pylist() == [1, None, 3]
    assert table.column("b").to_pylist()[0] == 4.0
    assert table.column("b").to_pylist()[1] == 5.0
    assert pd.isna(table.column("b").to_pylist()[2])


def test_write_parquet_fails_on_directory_path(tmp_path):
    """Verify that write_parquet raises an error if the output path is a directory."""
    # Arrange
    output_dir = tmp_path / "a_directory"
    output_dir.mkdir()
    schema = pa.schema([("a", pa.int32())])
    data = [{"a": 1}]

    # Act & Assert
    # We expect an IsADirectoryError or a similar FileExistsError on POSIX/Windows
    with pytest.raises(Exception) as excinfo:
        write_parquet(data, output_dir, schema)

    # Allow for different OS-specific errors
    assert isinstance(excinfo.value, (IOError, PermissionError))


def test_write_parquet_table_needs_cast(tmp_path):
    """Verify writing a table that requires schema casting succeeds."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    # Define a target schema with int64
    target_schema = pa.schema([pa.field("id", pa.int64())])
    # Create data with a schema that can be cast (int32)
    data_schema = pa.schema([pa.field("id", pa.int32())])
    data = pa.Table.from_pylist([{"id": 1}, {"id": 2}], schema=data_schema)

    # Act
    write_parquet(data, output_path, target_schema)

    # Assert
    assert output_path.exists()
    written_table = pq.read_table(output_path)
    # The written schema must match the target schema, not the source
    assert written_table.schema.equals(target_schema)
    assert written_table.num_rows == 2


def test_write_parquet_mkdir_os_error(tmp_path):
    """Verify that an OSError during directory creation is propagated."""
    # Arrange
    output_path = tmp_path / "nonexistent" / "test.parquet"
    schema = pa.schema([("a", pa.int32())])
    data = [{"a": 1}]

    # Act & Assert
    with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
        with pytest.raises(OSError, match="Permission denied"):
            write_parquet(data, output_path, schema)


@pytest.mark.parametrize(
    "unsupported_data",
    [
        {"a": 1},  # Raw dictionary, not in a list
        {1, 2, 3},  # Set
        "a string",  # Raw string
    ],
)
def test_write_parquet_unsupported_type_error(tmp_path, unsupported_data):
    """Verify that an unsupported data type raises a SchemaValidationError."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int32())])

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        write_parquet(unsupported_data, output_path, schema)


def test_write_parquet_table_different_column_order(tmp_path):
    """Verify that a table with a different column order is correctly written."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    target_schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("value", pa.string()),
        ]
    )
    # Create a table with columns in a different order
    data_schema = pa.schema(
        [
            pa.field("value", pa.string()),
            pa.field("id", pa.int64()),
        ]
    )
    data = pa.Table.from_pylist(
        [
            {"value": "a", "id": 1},
            {"value": "b", "id": 2},
        ],
        schema=data_schema,
    )

    # Act
    write_parquet(data, output_path, target_schema)

    # Assert
    assert output_path.exists()
    written_table = pq.read_table(output_path)
    assert written_table.schema.equals(target_schema)
    assert written_table.num_rows == 2


# region Edge Case Tests


@pytest.mark.parametrize(
    "empty_data",
    [
        pd.DataFrame({"id": [], "name": [], "value": []}),
        [],
        pa.Table.from_pylist([], schema=SCHEMA),
    ],
    ids=["empty_pandas_df", "empty_list", "empty_arrow_table"],
)
def test_write_parquet_empty_data(tmp_path: Path, empty_data):
    """Verifies that write_parquet correctly handles empty data inputs."""
    output_file = tmp_path / "test.parquet"
    write_parquet(empty_data, output_file, SCHEMA)

    assert output_file.exists()
    table = pa.parquet.read_table(output_file)
    assert table.schema.equals(SCHEMA, check_metadata=False)
    assert table.num_rows == 0


def test_write_parquet_preserves_target_schema_metadata(tmp_path: Path):
    """
    Verifies that the metadata from the target schema is applied, not the
    source data's metadata.
    """
    output_file = tmp_path / "test.parquet"
    source_metadata = {b"source": b"pandas_df"}
    target_metadata = {b"target": b"final_schema"}

    # Create a source table with metadata
    source_table = pa.Table.from_pandas(PANDAS_DF, preserve_index=False)
    source_table = source_table.replace_schema_metadata(source_metadata)

    # Define a target schema with different metadata
    target_schema = source_table.schema.with_metadata(target_metadata)

    write_parquet(source_table, output_file, target_schema)

    # Read the file and check its metadata
    written_table = pa.parquet.read_table(output_file)
    assert written_table.schema.metadata == target_metadata


# endregion

# region Negative Test Cases


def test_write_parquet_schema_validation_error(tmp_path: Path):
    """
    Verifies that write_parquet raises SchemaValidationError for incompatible data.
    """
    output_file = tmp_path / "test.parquet"
    # Data with a missing column 'id'
    bad_data = [{"name": "Eve", "value": 40.4}]

    with pytest.raises(SchemaValidationError):
        write_parquet(bad_data, output_file, SCHEMA)

    assert not output_file.exists()


def test_write_parquet_logs_error_on_cleanup_failure(tmp_path, mocker):
    """
    Verifies that if os.remove fails in the finally block, the error is logged.
    """
    output_path = tmp_path / "test.parquet"
    mock_logger_error = mocker.patch("py_parquet_forge.main.logger.error")

    # Simulate a failure during the os.replace to trigger the finally block
    with patch("os.replace", side_effect=IOError("Move failed")):
        # Also simulate a failure during the cleanup's os.remove
        with patch("os.remove", side_effect=OSError("Cleanup failed")):
            with pytest.raises(IOError, match="Move failed"):
                write_parquet(LIST_OF_DICTS, output_path, SCHEMA)

    # Check that the cleanup error was logged
    assert any(
        "Error removing temporary file" in call.args[0]
        for call in mock_logger_error.call_args_list
    )
    assert any(
        "Cleanup failed" in call.args[0] for call in mock_logger_error.call_args_list
    )


def test_write_parquet_logs_arrow_exception_on_write(tmp_path, mocker):
    """
    Verifies that an ArrowException during the write operation is logged and
    wrapped in SchemaValidationError.
    """
    output_path = tmp_path / "test.parquet"
    mock_logger_error = mocker.patch("py_parquet_forge.main.logger.error")

    with patch(
        "pyarrow.parquet.write_table", side_effect=pa.ArrowException("Write error")
    ):
        with pytest.raises(SchemaValidationError, match="Write error"):
            write_parquet(LIST_OF_DICTS, output_path, SCHEMA)

    mock_logger_error.assert_called_once()
    assert "Arrow schema validation error" in mock_logger_error.call_args[0][0]


def test_atomic_write_failure_does_not_affect_original_file(tmp_path: Path):
    """
    Verifies that a failed overwrite operation does not delete or corrupt
    the original file.
    """
    output_file = tmp_path / "test.parquet"

    # 1. Write an initial file successfully.
    initial_df = pd.DataFrame({"col1": [1, 2]})
    initial_schema = pa.schema([pa.field("col1", pa.int64())])
    write_parquet(initial_df, output_file, initial_schema)

    # Verify initial write
    assert output_file.exists()
    read_table_before = pq.read_table(output_file)
    pd.testing.assert_frame_equal(initial_df, read_table_before.to_pandas())

    # 2. Attempt to overwrite, but simulate a failure during the write.
    overwrite_df = pd.DataFrame({"col2": ["a", "b"]})
    overwrite_schema = pa.schema([pa.field("col2", pa.string())])

    with patch("pyarrow.parquet.write_table", side_effect=IOError("Disk full")):
        with pytest.raises(IOError, match="Disk full"):
            write_parquet(overwrite_df, output_file, overwrite_schema)

    # 3. Verify the original file is untouched and no temp files remain.
    assert output_file.exists(), "Original file was deleted"
    read_table_after = pq.read_table(output_file)
    pd.testing.assert_frame_equal(initial_df, read_table_after.to_pandas())

    temp_files = list(tmp_path.glob("*.tmp"))
    assert not temp_files, f"Temporary files found after failed overwrite: {temp_files}"


# endregion

# region Atomic Behavior Tests


def test_write_parquet_atomic_success(tmp_path: Path):
    """Verifies that write_parquet performs an atomic rename and cleans up temp files."""
    output_file = tmp_path / "test.parquet"

    with patch("os.replace", wraps=os.replace) as mock_replace:
        write_parquet(LIST_OF_DICTS, output_file, SCHEMA)

        # 1. Check that the atomic rename was called
        mock_replace.assert_called_once()

        # 2. Check that the final file exists
        assert output_file.exists()

        # 3. Check that no temporary files are left
        temp_files = list(tmp_path.glob("*.tmp"))
        assert not temp_files, f"Temporary files found: {temp_files}"


def test_write_parquet_propagates_original_error_on_cleanup_failure(tmp_path: Path):
    """
    Verifies that if both the rename and the cleanup fail, the original
    rename exception is propagated.
    """
    output_file = tmp_path / "test.parquet"

    # This side effect simulates the creation of the temp file by the real write_table
    def write_table_side_effect(table, path, **kwargs):
        Path(path).touch()

    # Simulate an error during os.replace, and a subsequent error during os.remove
    with (
        patch("pyarrow.parquet.write_table", side_effect=write_table_side_effect),
        patch("os.replace", side_effect=IOError("Move failed")) as mock_replace,
        patch("os.remove", side_effect=OSError("Cleanup failed")) as mock_remove,
    ):
        # We expect to catch the original IOError from the rename operation
        with pytest.raises(IOError, match="Move failed"):
            write_parquet(LIST_OF_DICTS, output_file, SCHEMA)

    mock_replace.assert_called_once()
    mock_remove.assert_called_once()
    assert not output_file.exists()


def test_write_parquet_atomic_failure_cleanup(tmp_path: Path):
    """Verifies that write_parquet cleans up the temporary file on write failure."""
    output_file = tmp_path / "test.parquet"

    # Simulate an error during the write operation
    with patch("pyarrow.parquet.write_table", side_effect=IOError("Disk full")):
        with pytest.raises(IOError):
            write_parquet(LIST_OF_DICTS, output_file, SCHEMA)

    # Check that no temporary files are left
    temp_files = list(tmp_path.glob("*.tmp"))
    assert not temp_files, f"Temporary files found after failure: {temp_files}"

    # Check that the original file was not created or modified
    assert not output_file.exists()


# endregion
