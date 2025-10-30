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
from unittest.mock import patch

import pyarrow as pa
import pytest

from py_parquet_forge.main import write_parquet


def test_write_parquet_os_replace_failure_cleanup(tmp_path: Path):
    """
    Verifies that write_parquet cleans up the temporary file if os.replace fails.
    """
    output_file = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    data = [{"a": 1}]

    # Simulate an OSError during the atomic move
    with patch(
        "py_parquet_forge.main.os.replace", side_effect=OSError("Permission denied")
    ) as mock_replace:
        with pytest.raises(OSError, match="Permission denied"):
            write_parquet(data, output_file, schema)

        # Verify that os.replace was actually attempted
        mock_replace.assert_called_once()

    # The most important check: The temporary file must be gone
    temp_files = list(tmp_path.glob("*.tmp"))
    assert not temp_files, f"Temporary file was not cleaned up: {temp_files}"

    # The final destination file should not have been created
    assert not output_file.exists()
