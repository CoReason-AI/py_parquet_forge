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

from py_parquet_forge.exceptions import SchemaValidationError
from py_parquet_forge.main import _convert_to_arrow_table


def test_convert_to_arrow_table_from_pydict_empty_list(tmp_path):
    """Verify that an empty list of dicts is correctly converted."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    data = []

    # Act
    table = _convert_to_arrow_table(data, schema)

    # Assert
    assert table.schema.equals(schema)
    assert table.num_rows == 0


def test_convert_to_arrow_table_from_record_batch(tmp_path):
    """Verify that a pyarrow.RecordBatch is correctly converted."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    record_batch = pa.RecordBatch.from_pydict({"a": [1, 2, 3]}, schema=schema)

    # Act
    table = _convert_to_arrow_table(record_batch, schema)

    # Assert
    assert table.schema.equals(schema)
    assert table.num_rows == 3


def test_convert_to_arrow_table_from_table(tmp_path):
    """Verify that a pyarrow.Table is correctly handled."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    arrow_table = pa.Table.from_pydict({"a": [1, 2, 3]}, schema=schema)

    # Act
    table = _convert_to_arrow_table(arrow_table, schema)

    # Assert
    assert table.schema.equals(schema)


def test_convert_to_arrow_table_unsupported_type(tmp_path):
    """Verify that a TypeError is raised for unsupported data types."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    data = "unsupported"

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        _convert_to_arrow_table(data, schema)


def test_convert_to_arrow_table_schema_already_correct(tmp_path):
    """Verify that no conversion is performed if the schema is already correct."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": [1, 2, 3]})

    # Act
    table = _convert_to_arrow_table(df, schema)

    # Assert
    assert table.schema.equals(schema)


def test_convert_to_arrow_table_schema_validation_error_missing_column(tmp_path):
    """Verify SchemaValidationError is raised when a column is missing."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32()), pa.field("b", pa.string())])
    df = pd.DataFrame({"a": [1, 2, 3]})  # Missing column "b"

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        _convert_to_arrow_table(df, schema)


def test_convert_to_arrow_table_from_empty_dataframe(tmp_path):
    """Verify that an empty pandas DataFrame is correctly converted."""
    # Arrange
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
        ]
    )
    df = pd.DataFrame({"id": [], "name": []})

    # Act
    table = _convert_to_arrow_table(df, schema)

    # Assert
    assert table.schema.equals(schema)
    assert table.num_rows == 0


def test_convert_to_arrow_table_with_all_null_values(tmp_path):
    """Verify that a DataFrame with all null values is handled correctly."""
    # Arrange
    schema = pa.schema(
        [
            pa.field("a", pa.int32()),
            pa.field("b", pa.float64()),
        ]
    )
    df = pd.DataFrame({"a": [None, None], "b": [pd.NA, pd.NA]})

    # Act
    table = _convert_to_arrow_table(df, schema)

    # Assert
    assert table.schema.equals(schema)
    assert table.num_rows == 2
    assert table.column("a").null_count == 2
    assert table.column("b").null_count == 2


def test_convert_to_arrow_table_ignores_extra_columns(tmp_path):
    """Verify that extra columns in the input data are correctly ignored."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],  # Extra column "b"
        }
    )

    # Act
    table = _convert_to_arrow_table(df, schema)

    # Assert
    assert table.schema.equals(schema)
    assert "b" not in table.schema.names


def test_convert_to_arrow_table_reorders_columns(tmp_path):
    """Verify that columns are correctly reordered to match the schema."""
    # Arrange
    schema = pa.schema([pa.field("b", pa.string()), pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    # Act
    table = _convert_to_arrow_table(df, schema)

    # Assert
    assert table.schema.equals(schema)
    assert table.column_names == ["b", "a"]


def test_convert_to_arrow_table_with_null_values(tmp_path):
    """Verify that None and NaN values are correctly handled."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32()), pa.field("b", pa.float64())])
    df = pd.DataFrame({"a": [1, None, 3], "b": [4.0, 5.0, pd.NA]})

    # Act
    table = _convert_to_arrow_table(df, schema)

    # Assert
    assert table.schema.equals(schema)
    assert table.column("a").to_pylist() == [1, None, 3]
    assert table.column("b").to_pylist()[0] == 4.0
    assert table.column("b").to_pylist()[1] == 5.0
    assert pd.isna(table.column("b").to_pylist()[2])


def test_convert_to_arrow_table_schema_validation_error(tmp_path):
    """Verify SchemaValidationError is raised for incompatible schemas."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": ["1", "2", "three"]})

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        _convert_to_arrow_table(df, schema)
