# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forget

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


def test_write_parquet_success_pandas(tmp_path):
    """Verify writing a pandas DataFrame to a Parquet file succeeds."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
        ]
    )
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }
    )

    # Act
    write_parquet(df, output_path, schema)

    # Assert
    assert output_path.exists()
    written_schema = pq.read_schema(output_path)
    assert written_schema.equals(schema)
    read_table = pq.read_table(output_path)
    assert read_table.num_rows == 3


def test_write_parquet_success_pydict(tmp_path):
    """Verify writing a list of dictionaries to a Parquet file succeeds."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("value", pa.float64()),
        ]
    )
    data = [
        {"id": 1, "value": 1.1},
        {"id": 2, "value": 2.2},
    ]

    # Act
    write_parquet(data, output_path, schema)

    # Assert
    assert output_path.exists()
    written_schema = pq.read_schema(output_path)
    assert written_schema.equals(schema)


def test_write_parquet_schema_validation_error(tmp_path):
    """Verify SchemaValidationError is raised for incompatible data."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    # Data has a string where an int is expected
    df = pd.DataFrame({"a": [1, 2, "not-an-int"]})

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        write_parquet(df, output_path, schema)

    # Assert that no file was created
    assert not output_path.exists()


def test_write_parquet_atomicity_on_failure(tmp_path):
    """Verify that no partial file is left if writing fails mid-way."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("col1", pa.int64())])
    df = pd.DataFrame({"col1": [1, 2, 3]})

    # Create a pre-existing file to ensure it's not touched on failure
    pre_existing_content = "pre-existing"
    output_path.write_text(pre_existing_content)

    # Mock pq.write_table to raise an exception during the write operation
    with patch("pyarrow.parquet.write_table", side_effect=IOError("Disk full!")):
        # Act & Assert
        with pytest.raises(IOError):
            write_parquet(df, output_path, schema)

    # Assert that the original file is untouched and no temp file exists
    assert output_path.read_text() == pre_existing_content

    # Check that no .tmp files are left in the directory
    temp_files = list(tmp_path.glob("*.tmp"))
    assert not temp_files, f"Temp files found: {temp_files}"


def test_write_parquet_os_error_on_cleanup(tmp_path):
    """Verify that an OSError during cleanup is logged but not propagated."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": [1]})

    # Mock os.replace to allow the temporary file to be created but not renamed
    with patch("os.replace"):
        # Mock os.remove to raise an OSError
        with patch("os.remove", side_effect=OSError("Permission denied")):
            # Act
            write_parquet(df, output_path, schema)

    # Assert
    # The test passes if no exception is raised


def test_write_parquet_exception_on_replace(tmp_path):
    """Verify that an exception during replace is handled correctly."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    df = pd.DataFrame({"a": [1]})

    # Mock os.replace to raise an exception
    with patch("os.replace", side_effect=Exception("Test exception")):
        # Act & Assert
        with pytest.raises(Exception):
            write_parquet(df, output_path, schema)


def test_write_parquet_overwrites_existing_file(tmp_path):
    """Verify that write_parquet overwrites an existing file."""
    # Arrange
    output_path = tmp_path / "test.parquet"

    # First write
    schema1 = pa.schema([pa.field("a", pa.int32())])
    df1 = pd.DataFrame({"a": [1]})
    write_parquet(df1, output_path, schema1)

    table1 = pq.read_table(output_path)
    assert table1.num_rows == 1
    assert table1.schema.equals(schema1)

    # Second write (overwrite)
    schema2 = pa.schema([pa.field("b", pa.string())])
    df2 = pd.DataFrame({"b": ["x", "y"]})

    # Act
    write_parquet(df2, output_path, schema2)

    # Assert
    assert output_path.exists()
    table2 = pq.read_table(output_path)
    assert table2.num_rows == 2
    assert table2.schema.equals(schema2)


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


def test_write_parquet_empty_pydict(tmp_path):
    """Verify that an empty list of dicts can be written to a Parquet file."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    data = []

    # Act
    write_parquet(data, output_path, schema)

    # Assert
    assert output_path.exists()
    written_schema = pq.read_schema(output_path)
    assert written_schema.equals(schema)
    read_table = pq.read_table(output_path)
    assert read_table.num_rows == 0


def test_write_parquet_path_with_spaces(tmp_path):
    """Verify writing to a path with spaces succeeds."""
    # Arrange
    output_path = tmp_path / "path with spaces" / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32())])
    data = [{"a": 1}]

    # Act
    write_parquet(data, output_path, schema)

    # Assert
    assert output_path.exists()
    read_table = pq.read_table(output_path)
    assert read_table.num_rows == 1


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


def test_write_parquet_with_na_values(tmp_path):
    """Verify that pd.NA values are correctly handled."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int32()), pa.field("b", pa.float64())])
    df = pd.DataFrame({"a": [1, pd.NA, 3], "b": [4.0, 5.0, pd.NA]})

    # Act
    write_parquet(df, output_path, schema)

    # Assert
    assert output_path.exists()
    table = pq.read_table(output_path)
    assert table.schema.equals(schema)
    assert table.column("a").to_pylist() == [1, None, 3]
    assert table.column("b").to_pylist()[0] == 4.0
    assert table.column("b").to_pylist()[1] == 5.0
    assert pd.isna(table.column("b").to_pylist()[2])


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


def test_write_parquet_success_recordbatch(tmp_path):
    """Verify writing a pyarrow.RecordBatch to a Parquet file succeeds."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("id", pa.int64())])
    data = pa.RecordBatch.from_pylist([{"id": 1}, {"id": 2}], schema=schema)

    # Act
    write_parquet(data, output_path, schema)

    # Assert
    assert output_path.exists()
    written_schema = pq.read_schema(output_path)
    assert written_schema.equals(schema)
    read_table = pq.read_table(output_path)
    assert read_table.num_rows == 2


def test_write_parquet_table_needs_cast(tmp_path):
    """Verify writing a table that requires schema casting succeeds."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    # Define a target schema with int64
    target_schema = pa.schema([pa.field("id", pa.int64())])
    # Create data with a schema that can be cast (int32)
    data_schema = pa.schema([pa.field("id", pa.int32())])
    data = pa.Table.from_pylist([{"id": 1}, {"id": 2}], schema=data_schema)

    # Act
    write_parquet(data, output_path, target_schema)

    # Assert
    assert output_path.exists()
    written_table = pq.read_table(output_path)
    # The written schema must match the target schema, not the source
    assert written_table.schema.equals(target_schema)
    assert written_table.num_rows == 2


def test_write_parquet_kwargs_pass_through(tmp_path):
    """Verify that kwargs are passed to the underlying pyarrow writer."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([pa.field("a", pa.int64())])
    data = [{"a": 1}]

    # Act: Write with statistics disabled
    write_parquet(data, output_path, schema, write_statistics=False)

    # Assert: No statistics should be present
    assert output_path.exists()
    with pq.ParquetFile(output_path) as parquet_file:
        metadata = parquet_file.metadata
        assert metadata.num_row_groups == 1
        column_chunk = metadata.row_group(0).column(0)
        assert column_chunk.statistics is None

    # Act: Write again with statistics enabled (default behavior)
    write_parquet(data, output_path, schema)  # Let it use the default

    # Assert: Statistics should now be present
    with pq.ParquetFile(output_path) as parquet_file_with_stats:
        metadata_with_stats = parquet_file_with_stats.metadata
        column_chunk_with_stats = metadata_with_stats.row_group(0).column(0)
        assert column_chunk_with_stats.statistics is not None
        assert column_chunk_with_stats.statistics.has_min_max


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


def test_write_parquet_mkdir_os_error(tmp_path):
    """Verify that an OSError during directory creation is propagated."""
    # Arrange
    output_path = tmp_path / "nonexistent" / "test.parquet"
    schema = pa.schema([("a", pa.int32())])
    data = [{"a": 1}]

    # Act & Assert
    with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
        with pytest.raises(OSError, match="Permission denied"):
            write_parquet(data, output_path, schema)


@pytest.mark.parametrize(
    "unsupported_data",
    [
        {"a": 1},  # Raw dictionary, not in a list
        {1, 2, 3},  # Set
        "a string",  # Raw string
    ],
)
def test_write_parquet_unsupported_type_error(tmp_path, unsupported_data):
    """Verify that an unsupported data type raises a SchemaValidationError."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("a", pa.int32())])

    # Act & Assert
    with pytest.raises(SchemaValidationError):
        write_parquet(unsupported_data, output_path, schema)


def test_write_parquet_table_different_column_order(tmp_path):
    """Verify that a table with a different column order is correctly written."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    target_schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("value", pa.string()),
        ]
    )
    # Create a table with columns in a different order
    data_schema = pa.schema(
        [
            pa.field("value", pa.string()),
            pa.field("id", pa.int64()),
        ]
    )
    data = pa.Table.from_pylist(
        [
            {"value": "a", "id": 1},
            {"value": "b", "id": 2},
        ],
        schema=data_schema,
    )

    # Act
    write_parquet(data, output_path, target_schema)

    # Assert
    assert output_path.exists()
    written_table = pq.read_table(output_path)
    assert written_table.schema.equals(target_schema)
    assert written_table.num_rows == 2


def test_write_parquet_success_table_matching_schema(tmp_path):
    """Verify writing a pyarrow.Table with a matching schema succeeds."""
    # Arrange
    output_path = tmp_path / "test.parquet"
    schema = pa.schema([("id", pa.int64())])
    data = pa.Table.from_pylist([{"id": 1}, {"id": 2}], schema=schema)

    # Act
    write_parquet(data, output_path, schema)

    # Assert
    assert output_path.exists()
    written_schema = pq.read_schema(output_path)
    assert written_schema.equals(schema)
    read_table = pq.read_table(output_path)
    assert read_table.num_rows == 2
