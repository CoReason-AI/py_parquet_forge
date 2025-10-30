# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from py_parquet_forge.exceptions import SchemaValidationError
from py_parquet_forge.main import (
    _convert_to_arrow_table,
    inspect_schema,
    read_parquet,
    read_parquet_iter,
    write_parquet,
    write_to_dataset,
)


def test_convert_to_arrow_table_inconsistent_dicts():
    """
    Edge Case: Verify that a list of dictionaries with inconsistent keys
    is correctly converted to a pyarrow.Table with nulls.
    """
    inconsistent_data = [
        {"id": 1, "name": "foo"},  # Missing 'value'
        {"id": 2, "name": "bar", "value": 2.2},
        {"id": 3, "value": 3.3},  # Missing 'name'
    ]
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("value", pa.float64()),
        ]
    )

    # Act
    table = _convert_to_arrow_table(inconsistent_data, schema)

    # Assert
    assert table.num_rows == 3
    assert table.schema.equals(schema, check_metadata=False)
    # Check that nulls were inserted correctly
    assert table.column("name")[0].as_py() == "foo"
    assert table.column("name")[2].as_py() is None
    assert table.column("value")[0].as_py() is None
    assert table.column("value")[2].as_py() == 3.3


def test_convert_to_arrow_table_non_finite_values():
    """
    Edge Case: Verify that converting a DataFrame with non-finite values
    (e.g., numpy.inf) to an integer schema raises a SchemaValidationError.
    """
    df_with_inf = pd.DataFrame({"value": [1.0, np.inf, -np.inf]})
    schema = pa.schema([pa.field("value", pa.int64())])

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        _convert_to_arrow_table(df_with_inf, schema)


def test_write_parquet_kwargs_passthrough(tmp_path):
    """
    Edge Case: Verify that kwargs are correctly passed to the underlying
    pyarrow writer by testing `write_statistics=False`.
    """
    output_path = tmp_path / "test_no_stats.parquet"
    schema = pa.schema([pa.field("a", pa.int64())])
    data = [{"a": 1}]

    # Act
    write_parquet(data, output_path, schema, write_statistics=False)

    # Assert
    assert output_path.exists()
    # Read the file's metadata and check the column chunk statistics
    parquet_file = pa.parquet.ParquetFile(output_path)
    # There should be one row group
    assert parquet_file.num_row_groups == 1
    row_group = parquet_file.metadata.row_group(0)
    # There should be one column chunk
    assert row_group.num_columns == 1
    column_chunk = row_group.column(0)
    # The statistics should be None because we disabled them
    assert column_chunk.statistics is None


def test_inspect_schema_non_parquet_file(tmp_path):
    """
    Failure Mode: Verify inspect_schema raises ArrowException for non-Parquet files.
    """
    non_parquet_file = tmp_path / "not_a_parquet.txt"
    non_parquet_file.write_text("This is not a parquet file")

    # Act & Assert
    with pytest.raises(pa.ArrowException):
        inspect_schema(non_parquet_file)


def test_inspect_schema_empty_directory(tmp_path):
    """
    Failure Mode: Verify inspect_schema raises ArrowException for an empty directory.
    """
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()

    # Act & Assert
    with pytest.raises(pa.ArrowException):
        inspect_schema(empty_dir)


def test_write_to_dataset_overwrite_with_empty_table(tmp_path):
    """
    Edge Case: Verify overwriting a dataset with an empty table deletes old data.
    """
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    # Create a dummy file to ensure it gets deleted
    (output_dir / "old_file.parquet").touch()

    schema = pa.schema([pa.field("a", pa.int64())])
    empty_data = []

    # Act
    write_to_dataset(empty_data, output_dir, schema, mode="overwrite")

    # Assert
    # The directory should exist, but the old file should be gone
    assert output_dir.exists()
    assert not (output_dir / "old_file.parquet").exists()
    # The directory should be empty of parquet files
    assert len(list(output_dir.glob("*.parquet"))) == 0


def test_read_parquet_mixed_type_nullable_integer(tmp_path):
    """
    Edge Case: Test `read_parquet` resilience with a mixed-type column
    that should be a nullable integer but contains non-numeric strings.
    """
    output_path = tmp_path / "mixed_type.parquet"
    # Create a table with an integer column that also contains a string
    # This can't be represented directly in Arrow, so we create it as a string column
    schema = pa.schema([pa.field("mixed_col", pa.string())])
    data = pa.Table.from_pydict(
        {"mixed_col": ["1", None, "not_a_number"]}, schema=schema
    )
    pa.parquet.write_table(data, output_path)

    # Act
    df = read_parquet(output_path)

    # Assert
    # The function should not raise an error.
    # The column should remain as object type due to the casting failure.
    assert df["mixed_col"].dtype == "object"
    assert df["mixed_col"][0] == "1"
    assert pd.isna(df["mixed_col"][1])
    assert df["mixed_col"][2] == "not_a_number"


def test_read_parquet_iter_zero_row_file(tmp_path):
    """
    Edge Case: Test `read_parquet_iter` on a dataset with zero-row files.
    """
    output_path = tmp_path / "zero_row.parquet"
    schema = pa.schema([pa.field("a", pa.int64())])
    empty_table = pa.Table.from_pylist([], schema=schema)
    pa.parquet.write_table(empty_table, output_path)

    # Act
    batches = list(read_parquet_iter(output_path))

    # Assert
    # pyarrow may yield a single, schema-only batch with zero rows
    # The important part is that the total number of rows is zero
    assert sum(batch.num_rows for batch in batches) == 0
