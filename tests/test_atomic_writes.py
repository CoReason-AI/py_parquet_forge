# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forget

import os
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest

from py_parquet_forge.main import write_parquet
from py_parquet_forge.exceptions import SchemaValidationError

TEST_SCHEMA = pa.schema([pa.field("id", pa.int64()), pa.field("name", pa.string())])
TEST_DATA = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


def test_write_parquet_atomic_success(tmp_path: Path):
    """Verifies that write_parquet performs an atomic rename and cleans up temp files."""
    output_file = tmp_path / "test.parquet"

    with patch("os.replace", wraps=os.replace) as mock_replace:
        write_parquet(TEST_DATA, output_file, TEST_SCHEMA)

        # 1. Check that the atomic rename was called
        mock_replace.assert_called_once()

        # 2. Check that the final file exists
        assert output_file.exists()

        # 3. Check that no temporary files are left
        temp_files = list(tmp_path.glob("*.tmp"))
        assert not temp_files, f"Temporary files found: {temp_files}"
