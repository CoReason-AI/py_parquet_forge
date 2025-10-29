# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forget

import pyarrow as pa
import pytest

from py_parquet_forge.exceptions import SchemaValidationError


def test_schema_validation_error_inheritance():
    """Verify that SchemaValidationError inherits from pyarrow.ArrowException."""
    assert issubclass(SchemaValidationError, pa.ArrowException)


def test_raise_schema_validation_error():
    """Verify that the custom exception can be raised."""
    with pytest.raises(SchemaValidationError) as excinfo:
        raise SchemaValidationError("This is a test error.")
    assert "This is a test error." in str(excinfo.value)
