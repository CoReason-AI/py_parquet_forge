# Copyright (c) 2025 CoReason, Inc.
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


def test_convert_to_arrow_table_schema_validation_error(tmp_path):
    """Verify SchemaValidationError is raised for incompatible schemas."""
    # Arrange
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": ["1", "2", "three"]})

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        _convert_to_arrow_table(df, schema)
