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
import pytest

from py_parquet_forge import write_parquet


def test_write_parquet_basic_success(tmp_path):
    """
    Verifies that `write_parquet` successfully writes a pandas DataFrame
    to a Parquet file with the correct schema and data.
    """
    # 1. Define schema and data
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("name", pa.string()),
            pa.field("value", pa.float64()),
        ]
    )
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [1.1, 2.2, 3.3],
        }
    )
    output_path = tmp_path / "test.parquet"

    # 2. Call the function to write the data
    write_parquet(data=df, output_path=output_path, schema=schema)

    # 3. Verify the output
    assert output_path.exists()

    # 4. Read the file back with pyarrow and validate
    table_read = pa.parquet.read_table(output_path)

    # Assert schema equality (including metadata)
    assert table_read.schema.equals(schema, check_metadata=True)

    # Assert data equality
    df_read = table_read.to_pandas()
    pd.testing.assert_frame_equal(df, df_read)


def test_write_parquet_atomic_failure(tmp_path, mocker):
    """
    Verifies that `write_parquet` does not leave a partial file and cleans up
    the temporary file if an error occurs during the write process.
    """
    schema = pa.schema([pa.field("a", pa.int64())])
    df = pd.DataFrame({"a": [1]})
    output_path = tmp_path / "test.parquet"

    # Create a pre-existing file to ensure it's not touched on failure.
    original_content = "pre-existing data"
    output_path.write_text(original_content)

    # Mock the underlying write_table function to simulate a failure.
    mocker.patch(
        "py_parquet_forge.main.pq.write_table",
        side_effect=OSError("Disk is full"),
    )

    # Expect the function to fail by raising the mocked exception.
    with pytest.raises(OSError, match="Disk is full"):
        write_parquet(data=df, output_path=output_path, schema=schema)

    # 1. Verify the original file is untouched.
    assert output_path.exists()
    assert output_path.read_text() == original_content

    # 2. Verify no temporary files are left behind.
    temp_files = list(tmp_path.glob("*.tmp"))
    assert not temp_files, f"Temporary files found: {temp_files}"
