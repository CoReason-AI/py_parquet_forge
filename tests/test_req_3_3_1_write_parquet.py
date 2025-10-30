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

import pandas as pd
import pyarrow as pa
import pytest

from py_parquet_forge import SchemaValidationError, write_parquet

# Define a consistent schema and some test data
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
