# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pytest

from py_parquet_forge import SchemaValidationError, write_parquet

# Define a consistent schema and some test data
SCHEMA = pa.schema([pa.field("id", pa.int64(), nullable=False)])
PANDAS_DF = pd.DataFrame({"id": [1, 2, 3]})


def test_write_parquet_atomic_failure_cleanup(tmp_path: Path):
    """
    Verifies that write_parquet cleans up the temporary file on write failure
    and does not create the final output file.
    """
    output_file = tmp_path / "test.parquet"
    # The temp file name is non-deterministic, so we need to find it
    temp_file_dir = output_file.parent

    with patch(
        "py_parquet_forge.main.pq.write_table", side_effect=IOError("Disk full")
    ):
        with pytest.raises(IOError):
            write_parquet(PANDAS_DF, output_file, SCHEMA)

    # Check that the final file was not created
    assert not output_file.exists()

    # Check that no temporary files are left in the directory
    temp_files = list(temp_file_dir.glob("*.tmp"))
    assert not temp_files, f"Temporary files left behind: {temp_files}"


def test_write_parquet_atomic_failure_preserves_existing_file(tmp_path: Path):
    """
    Verifies that if a write fails, an existing file at the destination path
    is not deleted or modified.
    """
    output_file = tmp_path / "test.parquet"
    # Create a dummy file at the destination
    original_content = "pre-existing content"
    output_file.write_text(original_content)

    with patch(
        "py_parquet_forge.main.pq.write_table", side_effect=IOError("Disk full")
    ):
        with pytest.raises(IOError):
            write_parquet(PANDAS_DF, output_file, SCHEMA)

    # Verify the original file is untouched
    assert output_file.exists()
    assert output_file.read_text() == original_content


def test_write_parquet_duplicate_columns_raises_error(tmp_path: Path):
    """
    Verifies that write_parquet raises a SchemaValidationError when the input
    data contains duplicate column names, which pyarrow cannot handle.
    """
    output_file = tmp_path / "test.parquet"
    # The target schema is valid, but the data is not.
    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.string()),
        ]
    )
    # pyarrow.Table.from_pylist will raise a pyarrow.ArrowInvalid error here,
    # which _convert_to_arrow_table should catch and re-raise as our custom exception.
    # Note: This is not a dict, but a list of key-value pairs to allow duplicates.
    bad_data = pd.DataFrame([[1, "a"], [2, "b"]], columns=["a", "a"])

    with pytest.raises(SchemaValidationError):
        write_parquet(bad_data, output_file, schema)

    assert not output_file.exists()
