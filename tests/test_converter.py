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


def test_convert_to_arrow_table_pydict_missing_column():
    """Verify SchemaValidationError for a pydict missing a required column."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32()), pa.field("b", pa.string())])
    data = [{"a": 1}, {"a": 2}]  # Missing column "b"

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        _convert_to_arrow_table(data, schema)


def test_convert_to_arrow_table_pydict_ignores_extra_columns():
    """Verify that extra columns in a pydict are correctly ignored."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    data = [
        {"a": 1, "b": "extra"},
        {"a": 2, "b": "extra"},
    ]

    # Act
    table = _convert_to_arrow_table(data, schema)

    # Assert
    assert table.schema.equals(schema)
    assert "b" not in table.schema.names


def test_convert_to_arrow_table_pydict_reorders_columns():
    """Verify that columns from a pydict are correctly reordered."""
    # Arrange
    schema = pa.schema([pa.field("b", pa.string()), pa.field("a", pa.int32())])
    data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    # Act
    table = _convert_to_arrow_table(data, schema)

    # Assert
    assert table.schema.equals(schema)
    assert table.column_names == ["b", "a"]


def test_convert_to_arrow_table_preserves_timestamp_precision():
    """
    Verifies that high-precision timestamps are preserved during conversion.
    """
    # Arrange
    ts_ns = pd.Timestamp("2023-01-01 12:34:56.789123456")
    df = pd.DataFrame({"time": [ts_ns]})

    # Target schema requires nanosecond precision
    schema = pa.schema([pa.field("time", pa.timestamp("ns"))])

    # Act
    table = _convert_to_arrow_table(df, schema)
    result_scalar = table.column("time")[0]

    # Assert
    assert table.schema.equals(schema)
    assert pa.types.is_timestamp(table.column("time").type)
    assert result_scalar.as_py() == ts_ns
    assert result_scalar.value % 1000 == 456  # Check nanosecond part


def test_convert_to_arrow_table_handles_pandas_nullable_int():
    """
    Verifies that pandas nullable integer types (Int64) are handled correctly,
    preserving nulls.
    """
    # Arrange
    # Use pandas' nullable integer type
    df = pd.DataFrame({"a": [1, pd.NA, 3]}, dtype="Int64")

    # Target schema is a standard pyarrow integer type
    schema = pa.schema([pa.field("a", pa.int64())])

    # Act
    table = _convert_to_arrow_table(df, schema)

    # Assert
    assert table.schema.equals(schema)
    assert table.column("a").null_count == 1
    assert table.column("a").to_pylist() == [1, None, 3]


def test_convert_to_arrow_table_preserves_schema_metadata():
    """
    Verifies that the custom metadata from the target schema is correctly applied
    to the final table, even when casting is performed.
    """
    # Arrange
    # Create a DataFrame. When converted to an Arrow Table, it will have pandas metadata.
    df = pd.DataFrame({"a": [1, 2, 3]})

    # Create a target schema with custom metadata that is different from pandas metadata.
    custom_metadata = {b"source": b"test_metadata", b"version": b"1"}
    target_schema = pa.schema([pa.field("a", pa.int64())]).with_metadata(
        custom_metadata
    )

    # Act
    # The function should see that the schemas are not equal (due to metadata and type diff)
    # and perform a cast, then replace the metadata.
    table = _convert_to_arrow_table(df, target_schema)

    # Assert
    # The final schema must be exactly equal to our target schema, including metadata.
    assert table.schema.equals(target_schema)
    assert table.schema.metadata == custom_metadata


def test_convert_to_arrow_table_handles_pandas_metadata_difference():
    """
    Verify that conversion succeeds even if the initial Arrow schema from pandas
    differs from the target schema only by metadata.
    """
    # Arrange
    # pa.Table.from_pandas adds {'pandas': ...} metadata to the schema
    df = pd.DataFrame({"a": [1, 2, 3]})
    # Our target schema is identical in structure but lacks the pandas metadata
    schema = pa.schema([pa.field("a", pa.int64())])

    # Act
    # The function should detect the schemas are not equal and proceed to cast,
    # resulting in a table with the target schema.
    table = _convert_to_arrow_table(df, schema)

    # Assert
    assert table.schema.equals(schema)


def test_convert_to_arrow_table_pydict_successful_cast():
    """
    Verify that a pydict is successfully cast to a different but compatible
    numeric type.
    """
    # Arrange
    schema = pa.schema([pa.field("a", pa.float64())])
    # pyarrow will infer int64 for this list
    data = [{"a": 1}, {"a": 2}, {"a": 3}]

    # Act
    table = _convert_to_arrow_table(data, schema)

    # Assert
    assert table.schema.equals(schema)
    assert pa.types.is_float64(table.column("a").type)
    assert table.column("a").to_pylist() == [1.0, 2.0, 3.0]


def test_convert_to_arrow_table_pydict_internal_type_error():
    """
    Verify SchemaValidationError is raised when a pydict contains a value
    that pyarrow cannot convert at all (e.g., a set), triggering an internal
    ArrowTypeError.
    """
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    # A 'set' is not a type that pyarrow can handle during conversion
    data = [{"a": 1}, {"a": {2, 3}}]

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        _convert_to_arrow_table(data, schema)


def test_convert_to_arrow_table_pydict_schema_validation_error(tmp_path):
    """
    Verify SchemaValidationError is raised for a pydict with an incompatible schema.
    """
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    data = [{"a": 1}, {"a": "two"}, {"a": 3}]  # Incompatible value "two"

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        _convert_to_arrow_table(data, schema)


def test_convert_to_arrow_table_mixed_type_object_column_error(tmp_path):
    """
    Verify SchemaValidationError is raised for a DataFrame with a mixed-type
    object column that cannot be cast to the target schema.
    """
    # Arrange
    schema = pa.schema([pa.field("a", pa.int64())])
    # This DataFrame column will have dtype 'object' due to mixed types.
    # pyarrow will fail when trying to convert "two" to an integer.
    df = pd.DataFrame({"a": [1, "two", 3]})

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
