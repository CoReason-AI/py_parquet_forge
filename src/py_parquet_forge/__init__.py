# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_parquet_forge

"""
A Python utility for standardizing high-performance serialization of in-memory data objects into local Apache Parquet files. Built on PyArrow, it provides memory-efficient streaming, atomic writes, and schema-enforced dataset management.
"""

__version__ = "0.1.0"
__author__ = "Gowtham A Rao"
__email__ = "gowtham.rao@coreason.ai"

from .exceptions import SchemaValidationError
from .main import inspect_schema, read_parquet, write_parquet

__all__ = ["write_parquet", "inspect_schema", "read_parquet", "SchemaValidationError"]
