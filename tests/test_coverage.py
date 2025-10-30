# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest  # noqa: F401

from py_parquet_forge.main import read_parquet


@patch("py_parquet_forge.main.pq.read_table")
def test_read_parquet_uncastable_integer(mock_read_table):
    """
    Covers the case where a table's schema claims a column is integer,
    but the `to_pandas()` conversion results in a float column with
    uncastable values (like infinity), triggering the `except` block
    during the cast to 'Int64'.
    """
    # 1. Define a schema that claims a column is an integer and has a null.
    # The null is required to enter the conditional block that does the casting.
    schema = pa.schema([pa.field("a", pa.int64())])

    # 2. Create the problematic DataFrame that we want `to_pandas()` to return.
    # This DataFrame has a float type because of np.nan and contains np.inf,
    # which will cause the `astype('Int64')` to fail.
    problematic_df = pd.DataFrame({"a": [np.inf, np.nan, 1.0]})

    # 3. Create a mock for the `pyarrow.Table` object.
    mock_arrow_table = MagicMock()

    # 4. Configure the mock table's attributes.
    #    - It needs a `schema` attribute that the function will check.
    #    - Its `to_pandas` method must return our problematic DataFrame.
    mock_arrow_table.schema = schema
    mock_arrow_table.to_pandas.return_value = problematic_df

    # 5. Configure the main mock for `pq.read_table` to return our mock table.
    mock_read_table.return_value = mock_arrow_table

    # 6. Call the function. Inside `read_parquet`:
    #    - It gets `mock_arrow_table` from the mock.
    #    - It checks the schema and sees `a` is an integer.
    #    - It calls `table.to_pandas()`, which our mock intercepts, returning `problematic_df`.
    #    - It sees the column has nulls.
    #    - It enters the `try` block and attempts `astype("Int64")`.
    #    - The cast fails because of `np.inf`.
    #    - The `except` block is executed, and it passes.
    df = read_parquet("dummy/path.parquet", output_format="pandas")

    # 7. Verify that the function handled the error gracefully.
    # The final DataFrame should be the one returned by our mocked `to_pandas`.
    pd.testing.assert_frame_equal(df, problematic_df)

    # 8. Explicitly check the dtype and values.
    assert df["a"].dtype == np.float64
    assert np.isinf(df["a"][0])
    assert pd.isna(df["a"][1])
    assert df["a"][2] == 1.0
