# `py_parquet_forge` Documentation

## 1. Introduction

`py_parquet_forge` is a high-performance Python utility for standardizing the serialization and deserialization of in-memory data to and from Apache Parquet files on a local filesystem. It is built on top of the powerful `pyarrow` library, providing a simplified, robust, and memory-efficient API for common data engineering tasks.

### 1.1. Core Features

- **Memory-Efficient Operations:** Process datasets larger than system memory using streaming and chunked reads/writes.
- **Strict Schema Enforcement:** Guarantee data integrity by enforcing a predefined `pyarrow.Schema` on all write operations.
- **Atomic File Writes:** Prevent corrupted or partially written files with atomic write-to-temp-and-rename operations.
- **Flexible Data Input:** Accepts data from common Python structures like Pandas DataFrames, lists of dictionaries, and Arrow Tables/RecordBatches.
- **Dataset Management:** Supports appending, overwriting, and partitioning data in a Hive-style directory structure.
- **High-Level API:** Offers a simple, intuitive API that abstracts away the complexities of `pyarrow`.

## 2. Installation

To install `py_parquet_forge`, you can use `pip`:

```bash
pip install py_parquet_forge
```

## 3. Core Concepts

### 3.1. Schema

The schema is the backbone of `py_parquet_forge`. Every write operation requires a `pyarrow.Schema` object, which defines the column names, data types, and nullability for the dataset. This strict enforcement prevents data type degradation and ensures data quality.

**Example: Creating a Schema**

```python
import pyarrow as pa

# Define the schema for your data
user_schema = pa.schema([
    pa.field("user_id", pa.int64(), nullable=False),
    pa.field("username", pa.string(), nullable=False),
    pa.field("signup_timestamp", pa.timestamp("us"), nullable=True),
    pa.field("is_active", pa.bool_(), nullable=False),
    pa.field("created_date", pa.date32(), nullable=False)
])
```

### 3.2. Input Data (`InputData`)

The package accepts data in several common formats:
- `pandas.DataFrame`
- `list[dict[str, any]]`
- `pyarrow.Table`
- `pyarrow.RecordBatch`

`py_parquet_forge` internally converts all these formats into a `pyarrow.Table` and validates it against the provided schema.

## 4. API Reference and Usage

### 4.1. Writing Data

#### `write_parquet`

For writing a complete, in-memory data object to a **single Parquet file atomically**. This is ideal for smaller datasets where atomicity is critical.

**Signature:**
```python
def write_parquet(
    data: InputData,
    output_path: PathLike,
    schema: PyArrowSchema,
    **kwargs
) -> None:
```

**Example:**
```python
import pandas as pd
from py_parquet_forge import write_parquet

# Sample data
df = pd.DataFrame({
    "user_id": [1, 2],
    "username": ["Alice", "Bob"],
    "signup_timestamp": [pd.Timestamp.now(), None],
    "is_active": [True, False],
    "created_date": [pd.to_datetime("2023-01-01").date()] * 2
})

# Write the DataFrame to a single file
write_parquet(df, "users.parquet", user_schema, compression="snappy")
```

#### `write_to_dataset`

For writing data to a **directory structure**, which can be partitioned. This function is used for managing larger datasets and for appending data over time.

**Signature:**
```python
def write_to_dataset(
    data: InputData,
    output_dir: PathLike,
    schema: PyArrowSchema,
    partition_cols: list[str] | None = None,
    mode: str = 'append',
    **kwargs
) -> None:
```

- `mode='append'`: Adds new Parquet files to the dataset.
- `mode='overwrite'`: Deletes the entire directory and writes new data.

**Example: Appending and Partitioning**
```python
from py_parquet_forge import write_to_dataset

# New data to append
new_users = [{"user_id": 3, "username": "Charlie", "is_active": True, "created_date": pd.to_datetime("2023-01-02").date()}]

# Append to a dataset partitioned by the 'created_date' column
write_to_dataset(
    new_users,
    "users_dataset",
    user_schema,
    partition_cols=["created_date"],
    mode="append"
)
```
This will create a directory structure like:
```
users_dataset/
└── created_date=2023-01-02/
    └── some-uuid.parquet
```

#### `ParquetStreamWriter`

A context manager for **streaming large datasets in chunks** to a single Parquet file. This is the most memory-efficient way to write a large file.

**Signature:**
```python
class ParquetStreamWriter:
    def __init__(self, output_path: PathLike, schema: PyArrowSchema, **kwargs)
    def write_chunk(self, data: InputData) -> None
```

**Example: Streaming Data**
```python
from py_parquet_forge import ParquetStreamWriter

# A generator function simulating a large data stream
def generate_large_data():
    for i in range(100):
        yield pd.DataFrame({
            "user_id": range(i * 1000, (i + 1) * 1000),
            "username": [f"user_{j}" for j in range(i * 1000, (i + 1) * 1000)],
            "signup_timestamp": [pd.Timestamp.now()] * 1000,
            "is_active": [True] * 1000,
            "created_date": [pd.to_datetime("2023-01-01").date()] * 1000
        })

# Use the stream writer to write chunks
with ParquetStreamWriter("large_users.parquet", user_schema) as writer:
    for chunk in generate_large_data():
        writer.write_chunk(chunk)
```

### 4.2. Reading Data

#### `read_parquet`

Reads an entire Parquet file or dataset into memory, returning a `pandas.DataFrame` or `pyarrow.Table`.

**Signature:**
```python
def read_parquet(
    input_path: PathLike,
    output_format: str = 'pandas',
    columns: list[str] | None = None,
    filters: PyArrowFilters | None = None,
    **kwargs
) -> pandas.DataFrame | pyarrow.Table:
```

- `columns`: A list of column names to read (column projection).
- `filters`: A `pyarrow.compute` expression to filter rows at the storage level (predicate pushdown).

**Example: Reading with Filters**
```python
from py_parquet_forge import read_parquet
import pyarrow.compute as pc

# Read only active users from the dataset
active_users_df = read_parquet(
    "users_dataset",
    columns=["user_id", "username"],
    filters=(pc.field("is_active") == True)
)
```

#### `read_parquet_iter`

A generator function for reading a large Parquet file or dataset in **memory-efficient chunks**.

**Signature:**
```python
def read_parquet_iter(
    input_path: PathLike,
    chunk_size: int = 100_000,
    columns: list[str] | None = None,
    filters: PyArrowFilters | None = None,
    **kwargs
) -> Iterator[pyarrow.RecordBatch]:
```

**Example: Processing a Large File in Batches**
```python
from py_parquet_forge import read_parquet_iter

# Iterate over record batches from the large file
total_users = 0
for batch in read_parquet_iter("large_users.parquet", chunk_size=10000):
    total_users += len(batch)

print(f"Total users processed: {total_users}")
```

### 4.3. Utilities

#### `inspect_schema`

Reads and returns the `pyarrow.Schema` of a Parquet file or dataset without loading the data into memory.

**Signature:**
```python
def inspect_schema(path: PathLike) -> PyArrowSchema:
```

**Example:**
```python
from py_parquet_forge import inspect_schema

# Inspect the schema of the dataset we created
schema = inspect_schema("users_dataset")
print(schema)
```

## 5. Error Handling

`py_parquet_forge` defines a custom exception for schema-related errors:

- `SchemaValidationError`: Raised when the input data cannot be safely conformed to the target schema. This can happen due to:
  - Missing columns
  - Data type incompatibility (e.g., trying to cast a string to an integer)
  - Null values in a column that is declared non-nullable

Other standard exceptions like `FileNotFoundError` or `pyarrow.ArrowInvalid` may be raised by the underlying `pyarrow` library.
