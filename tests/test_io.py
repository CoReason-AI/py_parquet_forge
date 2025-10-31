# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from py_parquet_forge.exceptions import SchemaValidationError
from py_parquet_forge.main import (
    inspect_schema,
    read_parquet,
    write_parquet,
    write_to_dataset,
)


def test_inspect_schema_single_file(tmp_path):
    """Verify that inspect_schema correctly reads the schema from a single Parquet file."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    expected_schema = pa.schema(
        [
            pa.field("col1", pa.int64()),
            pa.field("col2", pa.string()),
        ]
    )
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        }
    )
    table = pa.Table.from_pandas(df, schema=expected_schema)
    pq.write_table(table, output_path)

    # Act
    actual_schema = inspect_schema(output_path)

    # Assert
    assert actual_schema.equals(expected_schema)


def test_inspect_schema_nonexistent_path(tmp_path):
    """Verify that inspect_schema raises an exception for a nonexistent path."""
    # Arrange
    nonexistent_path = tmp_path / "nonexistent"

    # Act & Assert
    with pytest.raises(pa.ArrowIOError):
        inspect_schema(nonexistent_path)


def test_inspect_schema_invalid_file(tmp_path):
    """Verify that inspect_schema raises an exception for an invalid file type."""
    # Arrange
    invalid_file = tmp_path / "invalid.txt"
    with open(invalid_file, "w") as f:
        f.write("this is not a parquet file")

    # Act & Assert
    with pytest.raises(pa.ArrowInvalid):
        inspect_schema(invalid_file)


def test_inspect_schema_dataset_directory(tmp_path):
    """Verify that inspect_schema correctly reads the schema from a dataset directory."""
    # Arrange
    output_dir = tmp_path / "dataset"
    write_schema = pa.schema(
        [
            pa.field("col1", pa.int64()),
            pa.field("col2", pa.string()),
            pa.field("partition_col", pa.string()),
        ]
    )
    data = {
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"],
        "partition_col": ["one", "two", "one"],
    }
    table = pa.Table.from_pydict(data, schema=write_schema)
    pq.write_to_dataset(table, root_path=output_dir, partition_cols=["partition_col"])

    # Act
    actual_schema = inspect_schema(output_dir)

    # When a dataset is read, the partition column is dictionary-encoded by default.
    # We must construct the expected schema to reflect this for a valid comparison.
    expected_schema_after_read = pa.schema(
        [
            pa.field("col1", pa.int64()),
            pa.field("col2", pa.string()),
            pa.field(
                "partition_col", pa.dictionary(pa.int32(), pa.string(), ordered=False)
            ),
        ]
    )

    # Assert
    assert actual_schema.equals(expected_schema_after_read)


def test_write_to_dataset_append_mode(tmp_path):
    """Verify writing in append mode adds new data without removing existing data."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    df1 = pd.DataFrame({"col1": [1, 2]})
    df2 = pd.DataFrame({"col1": [3, 4]})

    # Act
    write_to_dataset(df1, output_dir, schema, mode="append")
    write_to_dataset(df2, output_dir, schema, mode="append")

    # Assert
    dataset = pq.ParquetDataset(output_dir)
    table = dataset.read()
    assert table.num_rows == 4
    assert sorted(table.column("col1").to_pylist()) == [1, 2, 3, 4]


def test_write_to_dataset_overwrite_mode(tmp_path):
    """Verify writing in overwrite mode removes existing data before writing new data."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    df1 = pd.DataFrame({"col1": [1, 2]})
    df2 = pd.DataFrame({"col1": [3, 4]})

    # Act
    write_to_dataset(df1, output_dir, schema, mode="append")  # Initial write
    write_to_dataset(df2, output_dir, schema, mode="overwrite")  # Overwrite

    # Assert
    dataset = pq.ParquetDataset(output_dir)
    table = dataset.read()
    assert table.num_rows == 2
    assert sorted(table.column("col1").to_pylist()) == [3, 4]


def test_write_to_dataset_with_partitioning(tmp_path):
    """Verify that data is correctly partitioned into subdirectories."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("value", pa.int64()), ("part", pa.string())])
    df = pd.DataFrame({"value": [1, 2, 3], "part": ["a", "b", "a"]})

    # Act
    write_to_dataset(df, output_dir, schema, partition_cols=["part"])

    # Assert
    part_a_path = output_dir / "part=a"
    part_b_path = output_dir / "part=b"
    assert part_a_path.is_dir()
    assert part_b_path.is_dir()

    # Verify content of partition 'a'
    table_a = pq.read_table(part_a_path)
    assert table_a.num_rows == 2
    assert sorted(table_a.column("value").to_pylist()) == [1, 3]

    # Verify content of partition 'b'
    table_b = pq.read_table(part_b_path)
    assert table_b.num_rows == 1
    assert table_b.column("value").to_pylist() == [2]


def test_write_to_dataset_schema_validation_error(tmp_path):
    """Verify that SchemaValidationError is raised for invalid data."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    df = pd.DataFrame({"col1": ["not-an-int"]})

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        write_to_dataset(df, output_dir, schema)
    assert not output_dir.exists()


def test_write_to_dataset_invalid_mode(tmp_path):
    """Verify that a ValueError is raised for an invalid mode."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    df = pd.DataFrame({"col1": [1]})

    # Act & Assert
    with pytest.raises(ValueError, match="mode must be either 'append' or 'overwrite'"):
        write_to_dataset(df, output_dir, schema, mode="invalid_mode")


def test_write_to_dataset_pydict_input(tmp_path):
    """Verify that a list of dictionaries can be written to a dataset."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("id", pa.int64())])
    data = [{"id": 1}, {"id": 2}]

    # Act
    write_to_dataset(data, output_dir, schema)

    # Assert
    table = pq.read_table(output_dir)
    assert table.num_rows == 2
    assert table.schema.equals(pa.schema([pa.field("id", pa.int64())]))


def test_write_to_dataset_empty_input(tmp_path):
    """Verify writing empty data creates the directory but no files."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    empty_df = pd.DataFrame({"col1": []})

    # Act
    write_to_dataset(empty_df, output_dir, schema)

    # Assert
    assert output_dir.is_dir()
    # Pyarrow does not write any files (including metadata) for an empty table
    assert not any(output_dir.iterdir())


def test_write_to_dataset_overwrite_os_error(tmp_path):
    """Verify that an OSError during directory removal is propagated."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("col1", pa.int64())])
    df = pd.DataFrame({"col1": [1]})
    write_to_dataset(df, output_dir, schema)  # Create the directory

    # Act & Assert
    with patch("shutil.rmtree", side_effect=OSError("Permission denied")):
        with pytest.raises(OSError, match="Permission denied"):
            write_to_dataset(df, output_dir, schema, mode="overwrite")


def test_write_to_dataset_invalid_partition_column(tmp_path):
    """Verify that an error is raised when a partition column does not exist."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("a", pa.int32())])
    data = [{"a": 1}]

    # Act & Assert
    with pytest.raises(
        pa.ArrowInvalid, match="Partition column 'non_existent_col' not in schema"
    ):
        write_to_dataset(data, output_dir, schema, partition_cols=["non_existent_col"])


def test_write_to_dataset_with_empty_partition_cols(tmp_path):
    """Verify that partition_cols=[] is treated as no partitioning."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("value", pa.int64()), ("part", pa.string())])
    df = pd.DataFrame({"value": [1, 2, 3], "part": ["a", "b", "a"]})

    # Act
    write_to_dataset(df, output_dir, schema, partition_cols=[])

    # Assert
    # Check that no partition subdirectories were created
    assert not any(f.is_dir() for f in output_dir.iterdir())

    # Check that at least one parquet file was created in the root
    parquet_files = list(output_dir.glob("*.parquet"))
    assert len(parquet_files) >= 1

    # Verify the content of the dataset
    table = pq.read_table(output_dir)
    assert table.num_rows == 3
    assert "part" in table.schema.names
    assert "value" in table.schema.names


def test_read_parquet_pandas_output(tmp_path):
    """Verify reading a Parquet file to a pandas DataFrame."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int64())])
    write_parquet([{"a": 1}], output_path, schema)

    # Act
    df = read_parquet(output_path, output_format="pandas")

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 1)
    assert df["a"][0] == 1


def test_read_parquet_arrow_output(tmp_path):
    """Verify reading a Parquet file to a pyarrow Table."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int64())])
    write_parquet([{"a": 1}], output_path, schema)

    # Act
    table = read_parquet(output_path, output_format="arrow")

    # Assert
    assert isinstance(table, pa.Table)
    assert table.num_rows == 1
    assert table.schema.equals(schema)


def test_read_parquet_with_column_projection(tmp_path):
    """Verify that only specified columns are read."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    write_parquet([{"a": 1, "b": "x"}], output_path, schema)

    # Act
    df = read_parquet(output_path, columns=["a"])

    # Assert
    assert "a" in df.columns
    assert "b" not in df.columns


def test_read_parquet_with_filters(tmp_path):
    """Verify that rows are filtered correctly."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("value", pa.int64())])
    data = [{"value": 10}, {"value": 20}, {"value": 30}]
    write_parquet(data, output_path, schema)

    # Act
    df = read_parquet(output_path, filters=[("value", ">", 15)])

    # Assert
    assert len(df) == 2
    assert sorted(df["value"].tolist()) == [20, 30]

    # Act
    table = read_parquet(
        output_path, filters=[("value", ">", 15)], output_format="arrow"
    )

    # Assert
    assert isinstance(table, pa.Table)
    assert table.num_rows == 2
    assert sorted(table.column("value").to_pylist()) == [20, 30]


def test_read_parquet_invalid_output_format(tmp_path):
    """Verify that a ValueError is raised for an invalid output format."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int64())])
    write_parquet([{"a": 1}], output_path, schema)

    # Act & Assert
    with pytest.raises(
        ValueError, match="output_format must be either 'pandas' or 'arrow'"
    ):
        read_parquet(output_path, output_format="invalid_format")


def test_read_parquet_from_dataset(tmp_path):
    """Verify reading from a partitioned dataset works correctly."""
    # Arrange
    output_dir = tmp_path / "dataset"
    schema = pa.schema([("value", pa.int64()), ("part", pa.string())])
    data = [
        {"value": 1, "part": "a"},
        {"value": 2, "part": "b"},
        {"value": 3, "part": "a"},
    ]
    write_to_dataset(data, output_dir, schema, partition_cols=["part"])

    # Act
    df = read_parquet(output_dir)

    # Assert
    assert len(df) == 3
    assert sorted(df["value"].tolist()) == [1, 2, 3]
    # In pyarrow datasets, partition columns are added as categorical
    assert df["part"].dtype == "category"


def test_write_to_dataset_mkdir_os_error(tmp_path):
    """Verify that an OSError during directory creation is propagated."""
    # Arrange
    output_dir = tmp_path / "nonexistent"
    schema = pa.schema([("a", pa.int32())])
    data = [{"a": 1}]

    # Act & Assert
    with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
        with pytest.raises(OSError, match="Permission denied"):
            write_to_dataset(data, output_dir, schema)


@pytest.mark.skipif(
    "sys.platform != 'win32'", reason="File locking is primarily a Windows concern"
)
def test_file_handle_is_released_after_inspect_schema(tmp_path):
    """
    Verify that inspect_schema releases its file handle, allowing the file
    to be immediately overwritten. This is critical on Windows.
    """
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int64())])
    write_parquet([{"a": 1}], output_path, schema)

    # Act
    # Inspect the schema, which on Windows could lock the file if not handled correctly.
    _ = inspect_schema(output_path)

    # Assert: The file should be immediately overwritable without a PermissionError
    try:
        write_parquet([{"a": 2}], output_path, schema)
    except PermissionError:
        pytest.fail(
            "PermissionError raised: inspect_schema did not release file handle."
        )
