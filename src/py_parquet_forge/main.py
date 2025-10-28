# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

import os
import uuid
from pathlib import Path
from typing import Any, TypeAlias, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from .exceptions import SchemaValidationError

# Define flexible type hints for function signatures
PathLike: TypeAlias = Union[str, os.PathLike, Path]
PyArrowSchema: TypeAlias = pa.Schema
InputData: TypeAlias = Union[
    list[dict[str, Any]], pd.DataFrame, pa.Table, pa.RecordBatch
]


def _convert_to_arrow_table(data: InputData, schema: PyArrowSchema) -> pa.Table:
    """Converts various Python data structures to a pyarrow.Table and validates the schema."""
    table: pa.Table

    try:
        if isinstance(data, pd.DataFrame):
            # Create table from pandas, letting pyarrow infer types initially.
            # This can fail on mixed-type object columns.
            table = pa.Table.from_pandas(data, preserve_index=False)
        elif isinstance(data, list):
            if not data:
                # Create an empty table with the provided schema.
                table = pa.Table.from_pylist([], schema=schema)
            else:
                # Create from list of dicts, inferring schema.
                table = pa.Table.from_pylist(data)
        elif isinstance(data, pa.RecordBatch):
            table = pa.Table.from_batches([data])
        elif isinstance(data, pa.Table):
            table = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # If the inferred schema already matches, we are done.
        if table.schema.equals(schema):
            return table

        # Reorder columns to match the target schema before casting.
        # A KeyError will be caught if a required column is missing.
        ordered_table = table.select([field.name for field in schema])

        # Cast to the final schema. This is the main validation step.
        # An ArrowInvalid will be caught if types are incompatible.
        return ordered_table.cast(target_schema=schema)

    except (pa.ArrowInvalid, KeyError, TypeError) as e:
        # Catch conversion/casting errors and raise our custom exception.
        raise SchemaValidationError(
            f"Failed to cast data to the target schema: {e}"
        ) from e


def write_parquet(
    data: InputData, output_path: PathLike, schema: PyArrowSchema, **kwargs: Any
) -> None:
    """
    Writes an in-memory data object to a single Parquet file atomically.

    This function ensures atomicity by first writing to a temporary file and then
    renaming it to the final destination. If any error occurs, the temporary
    file is cleaned up.

    :param data: The data to write (e.g., pandas.DataFrame, list of dicts).
    :param output_path: The destination file path.
    :param schema: The pyarrow.Schema to enforce.
    :param kwargs: Additional arguments passed to pyarrow.parquet.write_table.
    """
    table = _convert_to_arrow_table(data, schema)

    # Ensure the output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Create a temporary file path in the same directory
    temp_file_path = output_path_obj.with_suffix(f".{uuid.uuid4()}.tmp")

    try:
        pq.write_table(table, temp_file_path, **kwargs)
        # Atomically move the temporary file to the final destination, overwriting if it exists.
        # os.replace provides atomic overwrite functionality on both POSIX and Windows.
        os.replace(temp_file_path, output_path_obj)
        logger.info(f"Successfully wrote Parquet file to {output_path_obj}")
    except Exception as e:
        logger.error(f"Failed to write Parquet file to {output_path_obj}: {e}")
        # Re-raise the exception after attempting to clean up
        raise
    finally:
        # Clean up the temporary file if it still exists
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logger.debug(f"Removed temporary file {temp_file_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {temp_file_path}: {e}")


def inspect_schema(path: PathLike) -> pa.Schema:
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
