# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


def hello_world():
    logger.info("Hello World!")
    return "Hello World!"


def inspect_schema(path: str | Path) -> pa.Schema:
    """
    Reads the schema from a Parquet file or dataset.

    :param path: The path to the Parquet file or dataset directory.
    :return: The pyarrow.Schema object from the file/dataset metadata.
    """
    path_obj = Path(path)
    if path_obj.is_dir():
        dataset = pq.ParquetDataset(path_obj)
        return dataset.schema
    return pq.read_schema(path_obj)
