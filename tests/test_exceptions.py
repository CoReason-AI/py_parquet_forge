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
from py_parquet_forge.exceptions import SchemaValidationError


def test_schema_validation_error_inheritance():
    """Verify that SchemaValidationError inherits from pyarrow.ArrowException."""
    assert issubclass(SchemaValidationError, pa.ArrowException)


def test_raise_schema_validation_error():
    """Verify that the custom exception can be raised."""
    with pytest.raises(SchemaValidationError) as excinfo:
        raise SchemaValidationError("This is a test error.")
    assert "This is a test error." in str(excinfo.value)


def test_schema_validation_error_can_be_caught_as_arrow_exception():
    """Verify that SchemaValidationError can be caught via its base class."""
    try:
        raise SchemaValidationError("Test message")
    except pa.ArrowException as e:
        assert isinstance(e, SchemaValidationError)
        assert "Test message" in str(e)
    except Exception as e:
        pytest.fail(f"Exception was not caught as ArrowException, but as {type(e)}")


def test_write_parquet_schema_error_wraps_pyarrow_exception(tmp_path):
    """
    Verify that SchemaValidationError raised by write_parquet contains the
    underlying pyarrow exception as its cause.
    """
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    # This data will cause a pyarrow.ArrowInvalid error during casting
    df = pd.DataFrame({"a": ["not-a-number"]})

    # Act & Assert
    with pytest.raises(SchemaValidationError) as excinfo:
        write_parquet(df, output_path, schema)

    # Verify that the original pyarrow error is the cause
    assert isinstance(excinfo.value.__cause__, pa.ArrowInvalid)

    # Verify that the informative message from the original error is preserved
    # The exact message can vary between pyarrow versions, so we check for a key part
    assert "Failed to parse string" in str(excinfo.value)
