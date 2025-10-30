# Copyright (c) 2025 CoReason, Inc
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

        # If the inferred schema already matches (including metadata), we are done.
        if table.schema.equals(schema, check_metadata=True):
            return table

        # Reorder columns to match the target schema before casting.
        # A KeyError will be caught if a required column is missing.
        ordered_table = table.select([field.name for field in schema])

        # Cast to the final schema. This is the main validation step.
        # An ArrowInvalid will be caught if types are incompatible.
        casted_table = ordered_table.cast(target_schema=schema)

        # The cast operation may preserve the original table's metadata.
        # To ensure the final schema is exactly the one requested, we
        # explicitly apply the metadata from the target schema.
        return casted_table.replace_schema_metadata(schema.metadata)

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

    This function ensures that file handles are properly closed to avoid
    file locking issues, particularly on Windows.

    :param path: The path to the Parquet file or dataset directory.
    :return: The pyarrow.Schema object from the file/dataset metadata.
    """
    path_obj = Path(path)
    if path_obj.is_dir():
        # ParquetDataset handles its own resources
        dataset = pq.ParquetDataset(path_obj)
        return dataset.schema

    # For single files, use a context manager to ensure the file handle is released
    with pq.ParquetFile(path_obj) as parquet_file:
        return parquet_file.schema.to_arrow_schema()


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

    # Pre-validate that partition columns exist in the schema
    if partition_cols:
        schema_cols = set(table.schema.names)
        for col in partition_cols:
            if col not in schema_cols:
                raise pa.ArrowInvalid(f"Partition column '{col}' not in schema")

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
        df = table.to_pandas()
        # Manually convert integer columns with nulls to nullable integer types
        for field in table.schema:
            if pa.types.is_integer(field.type) and df[field.name].isnull().any():
                try:
                    df[field.name] = df[field.name].astype("Int64")
                except (TypeError, ValueError):
                    # This can happen if the column contains non-numeric data that
                    # couldn't be cast to a float. In this case, we leave it as is.
                    pass
        return df
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


class ParquetStreamWriter:
    """
    A context manager for writing large datasets in chunks to a single Parquet file.

    This class provides a memory-efficient way to serialize data by writing it
    in sequential chunks. It enforces a consistent schema across all chunks.

    Usage:
        with ParquetStreamWriter(output_path, schema) as writer:
            writer.write_chunk(chunk1)
            writer.write_chunk(chunk2)
    """

    def __init__(
        self, output_path: PathLike, schema: PyArrowSchema, **kwargs: Any
    ) -> None:
        """
        Initializes the ParquetStreamWriter.

        This will create a pyarrow.parquet.ParquetWriter and open the file for
        writing. The file at output_path will be overwritten if it already exists.

        :param output_path: The destination file path.
        :param schema: The pyarrow.Schema to enforce for all chunks.
        :param kwargs: Additional arguments passed to pyarrow.parquet.ParquetWriter.
        """
        self.schema = schema
        self._output_path = Path(output_path)
        self._writer_kwargs = kwargs
        self._writer: Optional[pq.ParquetWriter] = None

        # Ensure the output directory exists before opening the writer
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_chunk(self, data: InputData) -> None:
        """
        Writes a single chunk of data to the Parquet file.

        The data is converted to a pyarrow.Table and validated against the
        schema provided in the constructor before being written.

        :param data: The chunk of data to write.
        :raises SchemaValidationError: If the chunk's schema is incompatible.
        """
        if self._writer is None:
            raise IOError("Cannot write to a closed writer. Use within a 'with' block.")
        table = _convert_to_arrow_table(data, self.schema)
        self._writer.write_table(table)

    def __enter__(self) -> "ParquetStreamWriter":
        """Enters the context manager, opening the Parquet file for writing."""
        self._writer = pq.ParquetWriter(
            self._output_path, self.schema, **self._writer_kwargs
        )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Exits the context manager, ensuring the Parquet writer is closed.

        This finalizes the Parquet file, writing the necessary metadata and footer.
        It must be called to produce a valid file.
        """
        if self._writer:
            self._writer.close()
            self._writer = None

        # If an exception was raised within the 'with' block, clean up the created file.
        if exc_type is not None and self._output_path.exists():
            try:
                os.remove(self._output_path)
                logger.debug(
                    f"Removed partially written file {self._output_path} due to an exception."
                )
            except OSError as e:
                # Log the error, but don't re-raise, as the original exception is more important.
                logger.error(
                    f"Error removing partially written file {self._output_path}: {e}"
                )
