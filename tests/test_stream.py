# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forget

from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from py_parquet_forge import ParquetStreamWriter, SchemaValidationError


def test_stream_writer_success_pandas(tmp_path):
    """Verify writing a pandas DataFrame chunk succeeds."""
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
    with ParquetStreamWriter(output_path, schema) as writer:
        writer.write_chunk(df)

    # Assert
    assert output_path.exists()
    written_table = pq.read_table(output_path)
    assert written_table.schema.equals(schema)
    assert written_table.num_rows == 3


def test_stream_writer_schema_validation_error(tmp_path):
    """Verify SchemaValidationError is raised for an incompatible chunk."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": [1, 2, "not-an-int"]})

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        with ParquetStreamWriter(output_path, schema) as writer:
            writer.write_chunk(df)

    # Assert that no file was created
    assert not output_path.exists()


def test_stream_writer_multiple_chunks(tmp_path):
    """Verify writing multiple chunks combines them correctly in the final file."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("value", pa.int64())])
    data1 = [{"value": 1}, {"value": 2}]
    data2 = pd.DataFrame({"value": [3, 4, 5]})
    data3 = pa.Table.from_pydict({"value": [6]}, schema=schema)

    # Act
    with ParquetStreamWriter(output_path, schema) as writer:
        writer.write_chunk(data1)
        writer.write_chunk(data2)
        writer.write_chunk(data3)

    # Assert
    assert output_path.exists()
    written_table = pq.read_table(output_path)
    assert written_table.num_rows == 6
    assert written_table.column("value").to_pylist() == [1, 2, 3, 4, 5, 6]


def test_stream_writer_kwargs_passthrough(tmp_path):
    """Verify that kwargs are correctly passed to the underlying writer."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    data = [{"a": 1}]

    # Act
    # We test the kwarg passthrough by disabling statistics and checking the result.
    with ParquetStreamWriter(
        output_path, schema, compression="snappy", write_statistics=False
    ) as writer:
        writer.write_chunk(data)

    # Assert
    assert output_path.exists()
    file_metadata = pq.read_metadata(output_path)
    # Check that another kwarg (compression) was still applied correctly
    assert file_metadata.row_group(0).column(0).compression == "SNAPPY"
    # Check that statistics were not written, as requested by the kwarg
    assert file_metadata.row_group(0).column(0).statistics is None


def test_stream_writer_exception_handling_cleanup(tmp_path):
    """Verify the output file is removed if an exception occurs after the first chunk."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    data1 = [{"a": 1}]

    # Act & Assert
    with pytest.raises(ValueError, match="Test error"):
        with ParquetStreamWriter(output_path, schema) as writer:
            writer.write_chunk(data1)
            # This should write the first chunk successfully
            raise ValueError("Test error")

    # Assert that the file was cleaned up
    assert not output_path.exists()


def test_stream_writer_cleanup_os_error_on_exception(tmp_path):
    """Verify that an OSError during cleanup is logged but the original error is propagated."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    data = [{"a": 1}]
    original_error = ValueError("Original error inside with block")
    cleanup_error = OSError("Permission denied during cleanup")

    # Act & Assert
    with patch("os.remove", side_effect=cleanup_error):
        with pytest.raises(ValueError, match="Original error inside with block") as excinfo:
            with ParquetStreamWriter(output_path, schema) as writer:
                writer.write_chunk(data)
                raise original_error

    # Verify that the original exception was propagated, not the cleanup OSError.
    assert excinfo.value is original_error

    # The file may still exist since the cleanup failed, so we can check for that.
    assert output_path.exists()
