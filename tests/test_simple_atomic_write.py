# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

import pyarrow as pa
import pyarrow.parquet as pq
import pytest  # noqa: F401

from py_parquet_forge.main import write_parquet


def test_write_parquet_kwargs_passthrough(tmp_path):
    """
    Verifies that **kwargs are correctly passed through to the underlying
    pyarrow.parquet.write_table function.
    """
    schema = pa.schema([pa.field("a", pa.int64())])
    table = pa.Table.from_pydict({"a": [1, 2, 3]}, schema=schema)
    output_path = tmp_path / "test.parquet"

    # Use a kwarg that has a verifiable side-effect, like write_statistics.
    write_parquet(table, output_path, schema, write_statistics=False)

    # Inspect the written file's metadata.
    parquet_file = pq.ParquetFile(output_path)
    metadata = parquet_file.metadata

    # Check that statistics are not written for the column chunk.
    # If `write_statistics=False` was successful, the `statistics` attribute
    # on the column chunk metadata will be None.
    column_chunk = metadata.row_group(0).column(0)
    assert column_chunk.statistics is None
