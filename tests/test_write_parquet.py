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

from py_parquet_forge import SchemaValidationError, write_parquet

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
