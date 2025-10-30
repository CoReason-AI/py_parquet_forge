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
import pyarrow.parquet as pq
import pytest

from py_parquet_forge.exceptions import SchemaValidationError
from py_parquet_forge.main import write_parquet

# Define a consistent schema for all tests in this module
TEST_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("value", pa.float64()),
    ]
)

# Define sample data conforming to the schema
PANDAS_DF = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["foo", "bar", "baz"],
        "value": [1.1, 2.2, 3.3],
    }
)

LIST_OF_DICTS = PANDAS_DF.to_dict("records")
ARROW_TABLE = pa.Table.from_pandas(PANDAS_DF, schema=TEST_SCHEMA)


@pytest.mark.parametrize(
    "input_data,data_label",
    [
        (PANDAS_DF, "pandas_dataframe"),
        (LIST_OF_DICTS, "list_of_dicts"),
        (ARROW_TABLE, "arrow_table"),
    ],
)
def test_write_parquet_successful_atomic_write(
    tmp_path, input_data, data_label
):
    """
    REQ-3.3.1: Verify `write_parquet` successfully writes data from various
    InputData types, and the data read back is identical to the source.
    """
    output_path = tmp_path / f"test_output_{data_label}.parquet"

    # Act: Write the data to a Parquet file
    write_parquet(data=input_data, output_path=output_path, schema=TEST_SCHEMA)

    # Assert: Check that the file was created and the content is correct
    assert output_path.exists()

    # Read the data back using pyarrow directly to avoid test dependency
    read_table = pq.read_table(output_path)
    read_df = read_table.to_pandas()

    # Verify the schema of the written file is correct
    assert read_table.schema.equals(TEST_SCHEMA, check_metadata=False)

    # For comparison, ensure the read DataFrame matches the original pandas DataFrame
    expected_df = PANDAS_DF.astype({"id": "int64"})
    pd.testing.assert_frame_equal(read_df, expected_df)


def test_write_parquet_schema_validation_error(tmp_path):
    """
    REQ-6.1: Verify `write_parquet` raises SchemaValidationError for incompatible data.
    """
    output_path = tmp_path / "test_output_invalid.parquet"
    # Data with a missing column ('value')
    invalid_data = [{"id": 1, "name": "bad_data"}]

    # Act & Assert: Expect a SchemaValidationError
    with pytest.raises(SchemaValidationError):
        write_parquet(data=invalid_data, output_path=output_path, schema=TEST_SCHEMA)

    # Also assert that no partial/invalid file was left behind
    assert not output_path.exists()


def test_write_parquet_empty_data(tmp_path):
    """
    Boundary Case: Verify `write_parquet` can handle empty inputs correctly.
    """
    output_path = tmp_path / "test_output_empty.parquet"
    empty_data = []  # Empty list of dicts

    # Act
    write_parquet(data=empty_data, output_path=output_path, schema=TEST_SCHEMA)

    # Assert
    assert output_path.exists()
    read_table = pq.read_table(output_path)
    assert read_table.schema.equals(TEST_SCHEMA, check_metadata=False)
    assert read_table.num_rows == 0


def test_write_parquet_preserves_metadata(tmp_path):
    """
    Verify that custom metadata in the schema is correctly written to the file.
    """
    output_path = tmp_path / "test_output_metadata.parquet"
    custom_metadata = {b"source": b"test-suite", b"version": b"1.2.3"}
    schema_with_metadata = TEST_SCHEMA.with_metadata(custom_metadata)

    # Act
    write_parquet(
        data=PANDAS_DF, output_path=output_path, schema=schema_with_metadata
    )

    # Assert
    assert output_path.exists()
    written_schema = pq.read_schema(output_path)
    assert written_schema.metadata == custom_metadata
    # Use .equals with check_metadata=True for a full comparison
    assert written_schema.equals(schema_with_metadata, check_metadata=True)
