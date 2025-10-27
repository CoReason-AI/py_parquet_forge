# py-parquet-forge

A Python utility for standardizing high-performance serialization of in-memory data objects into local Apache Parquet files. Built on PyArrow, it provides memory-efficient streaming, atomic writes, and schema-enforced dataset management.

[![CI](https://github.com/CoReason-AI/py_parquet_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/py_parquet_forge/actions/workflows/ci.yml)

## Getting Started

### Prerequisites

- Python 3.10+
- Poetry

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/example/example.git
    cd my_python_project
    ```
2.  Install dependencies:
    ```sh
    poetry install
    ```

### Usage

-   Run the linter:
    ```sh
    poetry run pre-commit run --all-files
    ```
-   Run the tests:
    ```sh
    poetry run pytest
    ```
