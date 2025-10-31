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

# A consistent schema to be used for most tests
TARGET_SCHEMA = pa.schema(
    [
        pa.field("a", pa.int64(), nullable=False),
        pa.field("b", pa.string(), nullable=True),
    ],
    metadata={b"key": b"value"},
)


def test_convert_from_pandas_dataframe():
    """Tests conversion from a pandas DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    table = _convert_to_arrow_table(df, TARGET_SCHEMA)
    assert table.schema.equals(TARGET_SCHEMA, check_metadata=True)
    assert table.num_rows == 3


def test_convert_from_list_of_dicts():
    """Tests conversion from a list of dictionaries."""
    data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    table = _convert_to_arrow_table(data, TARGET_SCHEMA)
    assert table.schema.equals(TARGET_SCHEMA, check_metadata=True)
    assert table.num_rows == 2


def test_convert_from_pyarrow_record_batch():
    """Tests conversion from a PyArrow RecordBatch."""
    data = [pa.record_batch([[1, 2], ["x", "y"]], schema=TARGET_SCHEMA)]
    # This creates a table from a list of batches
    record_batch = pa.Table.from_batches(data).to_batches()[0]
    table = _convert_to_arrow_table(record_batch, TARGET_SCHEMA)
    assert table.schema.equals(TARGET_SCHEMA, check_metadata=True)
    assert table.num_rows == 2


def test_convert_from_pyarrow_table():
    """Tests that a PyArrow Table passes through correctly."""
    source_table = pa.Table.from_pydict({"a": [1], "b": ["x"]}, schema=TARGET_SCHEMA)
    table = _convert_to_arrow_table(source_table, TARGET_SCHEMA)
    assert table.schema.equals(TARGET_SCHEMA, check_metadata=True)
    assert table.num_rows == 1


def test_column_reordering():
    """Tests that columns are reordered to match the schema."""
    df = pd.DataFrame({"b": ["x", "y"], "a": [1, 2]})
    table = _convert_to_arrow_table(df, TARGET_SCHEMA)
    assert table.schema.equals(TARGET_SCHEMA, check_metadata=True)
    assert table.column_names == ["a", "b"]


def test_type_casting():
    """Tests that data types are cast correctly."""
    # Input has int32, but schema requires int64
    df = pd.DataFrame({"a": pd.Series([1, 2], dtype="int32"), "b": ["x", "y"]})
    table = _convert_to_arrow_table(df, TARGET_SCHEMA)
    assert table.schema.equals(TARGET_SCHEMA, check_metadata=True)
    assert table.schema.field("a").type == pa.int64()


def test_nullability_error():
    """Tests that nulls in a non-nullable column raise an error."""
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
    with pytest.raises(
        SchemaValidationError,
        match="Column 'a' is declared non-nullable but contains nulls",
    ):
        _convert_to_arrow_table(df, TARGET_SCHEMA)


def test_nullability_allowed():
    """Tests that nulls are allowed in a nullable column."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})
    table = _convert_to_arrow_table(df, TARGET_SCHEMA)
    assert table.schema.equals(TARGET_SCHEMA, check_metadata=True)
    assert table.column("b").null_count == 1


def test_unsupported_data_type():
    """Tests that an unsupported data type raises a TypeError."""
    data = {1, 2, 3}  # A set is not a supported type
    with pytest.raises(SchemaValidationError, match="Unsupported data type"):
        _convert_to_arrow_table(data, TARGET_SCHEMA)


def test_missing_column_in_list_of_dicts():
    """Tests that a missing non-nullable column in a list of dicts raises an error."""
    data = [{"a": 1, "b": "x"}, {"b": "y"}]  # Missing non-nullable 'a'
    with pytest.raises(SchemaValidationError, match="non-nullable but contains nulls"):
        _convert_to_arrow_table(data, TARGET_SCHEMA)


def test_metadata_is_applied():
    """Tests that metadata from the target schema is correctly applied."""
    # Create a source table without the target metadata.
    # Using from_pydict avoids adding pandas-specific metadata.
    source_table = pa.Table.from_pydict(
        {"a": [1], "b": ["x"]},
        schema=pa.schema(
            [
                pa.field("a", pa.int64(), nullable=False),
                pa.field("b", pa.string(), nullable=True),
            ]
        ),
    )

    # Ensure the table starts with no metadata
    assert not source_table.schema.metadata

    # After conversion, it should have the target schema's metadata
    converted_table = _convert_to_arrow_table(source_table, TARGET_SCHEMA)
    assert converted_table.schema.metadata
    assert converted_table.schema.metadata == {b"key": b"value"}


def test_explicit_null_check_is_triggered():
    """
    Tests that the explicit null check is triggered, not just the cast error.

    This test creates a scenario where the data type of a column is already
    correct (int64), but it contains nulls where the target schema forbids them.
    This bypasses errors during casting and isolates the explicit null check.
    """
    # Schema with a nullable integer column
    source_schema = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=True),
            pa.field("b", pa.string(), nullable=True),
        ]
    )
    # Target schema where the integer column is non-nullable
    target_schema = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=False),
            pa.field("b", pa.string(), nullable=True),
        ]
    )

    # Create a table with a null in the 'a' column
    source_table = pa.Table.from_pydict(
        {"a": [1, None, 3], "b": ["x", "y", "z"]}, schema=source_schema
    )

    # Expect our custom validation error with the specific message
    with pytest.raises(
        SchemaValidationError,
        match="Column 'a' is declared non-nullable but contains nulls",
    ):
        _convert_to_arrow_table(source_table, target_schema)
