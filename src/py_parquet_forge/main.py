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
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterator, List, Optional, TypeAlias, Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from loguru import logger

from .exceptions import SchemaValidationError

# Define flexible type hints for function signatures
PathLike: TypeAlias = Union[str, os.PathLike, Path]
PyArrowSchema: TypeAlias = pa.Schema
InputData: TypeAlias = Union[
    list[dict[str, Any]], pd.DataFrame, pa.Table, pa.RecordBatch
]
PyArrowFilters: TypeAlias = Any


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


def write_to_dataset(
    data: InputData,
    output_dir: PathLike,
    schema: PyArrowSchema,
    partition_cols: Optional[List[str]] = None,
    mode: str = "append",
    **kwargs: Any,
) -> None:
    """
    Writes a complete in-memory data object to a partitioned Parquet dataset.

    This function supports appending data to an existing dataset or overwriting it.
    It provides schema enforcement and partitioning capabilities.

    :param data: The data to write (e.g., pandas.DataFrame, list of dicts).
    :param output_dir: The root directory of the dataset.
    :param schema: The pyarrow.Schema to enforce.
    :param partition_cols: A list of column names to partition the data by.
    :param mode: Either 'append' (adds new files, default) or 'overwrite'
                 (deletes existing dataset content before writing).
    :param kwargs: Additional arguments passed to pyarrow.parquet.write_to_dataset.
    """
    if mode not in ["append", "overwrite"]:
        raise ValueError("mode must be either 'append' or 'overwrite'")

    output_dir_obj = Path(output_dir)

    if output_dir_obj.exists() and mode == "overwrite":
        try:
            shutil.rmtree(output_dir_obj)
            logger.info(f"Overwrite mode: Removed existing dataset at {output_dir_obj}")
        except OSError as e:
            logger.error(f"Error removing directory {output_dir_obj}: {e}")
            raise

    table = _convert_to_arrow_table(data, schema)

    # Only create the directory after schema validation has passed
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    pq.write_to_dataset(
        table,
        root_path=output_dir_obj,
        partition_cols=partition_cols or [],
        **kwargs,
    )
    logger.info(f"Successfully wrote data to dataset at {output_dir_obj}")


def read_parquet(
    input_path: PathLike,
    output_format: str = "pandas",
    columns: Optional[List[str]] = None,
    filters: Optional[PyArrowFilters] = None,
    **kwargs: Any,
) -> Union[pd.DataFrame, pa.Table]:
    """
    Reads an entire Parquet file or dataset into memory.

    This function provides a convenient way to load Parquet data, supporting
    column projection, predicate pushdown (filtering), and multiple output formats.

    :param input_path: The path to the Parquet file or dataset directory.
    :param output_format: The desired output format ('pandas' or 'arrow').
    :param columns: A list of column names to read (projection).
    :param filters: A PyArrow-compatible filter expression for predicate pushdown.
    :param kwargs: Additional arguments passed to pyarrow.parquet.read_table.
    :return: The data as a pandas.DataFrame or pyarrow.Table.
    """
    if output_format not in ["pandas", "arrow"]:
        raise ValueError("output_format must be either 'pandas' or 'arrow'")

    table = pq.read_table(input_path, columns=columns, filters=filters, **kwargs)

    if output_format == "pandas":
        return table.to_pandas()
    return table


def read_parquet_iter(
    input_path: PathLike,
    chunk_size: int = 100_000,
    columns: Optional[List[str]] = None,
    filters: Optional[PyArrowFilters] = None,
    **kwargs: Any,
) -> Iterator[pa.RecordBatch]:
    """
    Reads a large Parquet file or dataset in memory-efficient chunks.

    This function acts as a generator, yielding data in pyarrow.RecordBatch objects,
    which helps in processing datasets that are larger than available memory.

    :param input_path: The path to the Parquet file or dataset directory.
    :param chunk_size: The desired number of rows per chunk (approximate).
    :param columns: A list of column names to read (projection).
    :param filters: A PyArrow-compatible filter expression for predicate pushdown.
    :param kwargs: Additional arguments passed to the underlying pyarrow functions.
    :return: An iterator of pyarrow.RecordBatch objects.
    """
    # Use the `pyarrow.dataset` module for robust scanning
    dataset = ds.dataset(input_path, format="parquet", partitioning="hive", **kwargs)

    # The scanner provides fine-grained control over reading
    scanner = dataset.scanner(
        columns=columns,
        filter=filters,
        batch_size=chunk_size,
    )

    # Iterate over the scanner to yield record batches
    for batch in scanner.to_reader():
        yield batch
