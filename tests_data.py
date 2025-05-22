import pytest
import os
from pathlib import Path
from data import convert_one_file, TARGET_COLUMNS, CSV_TO_TARGET
import polars as pl
import csv
import logging

def test_convert_one_file_success(tmp_path):
    # Create a valid CSV file with all required columns
    csv_path = tmp_path / "test.csv"
    parquet_path = tmp_path / "test.parquet"
    header = list(CSV_TO_TARGET.keys())
    row = ["ABC123", "2024-01-01T12:00:00", "116.4", "39.9", "50.0", "1"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
    res = convert_one_file((str(csv_path), str(parquet_path), "zstd"))
    assert res['ok']
    assert os.path.exists(parquet_path)
    # Check Parquet content
    df = pl.read_parquet(parquet_path)
    assert set(df.columns) == set(TARGET_COLUMNS)
    assert df.shape[0] == 1

def test_convert_one_file_missing_column(tmp_path):
    # Create a CSV missing a required column
    csv_path = tmp_path / "test_missing.csv"
    parquet_path = tmp_path / "test_missing.parquet"
    header = list(CSV_TO_TARGET.keys())[:-1]  # Remove last required column
    row = ["ABC123", "2024-01-01T12:00:00", "116.4", "39.9", "50.0"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
    res = convert_one_file((str(csv_path), str(parquet_path), "zstd"))
    assert not res['ok']
    assert 'Missing required columns' in res['error']
    assert not os.path.exists(parquet_path)

def test_convert_one_file_dtype_enforcement(tmp_path):
    # Create a CSV with wrong dtype (e.g., string in a float column)
    csv_path = tmp_path / "test_dtype.csv"
    parquet_path = tmp_path / "test_dtype.parquet"
    header = list(CSV_TO_TARGET.keys())
    # instant_speed should be float, but put a string
    row = ["ABC123", "2024-01-01T12:00:00", "116.4", "39.9", "not_a_float", "1"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
    res = convert_one_file((str(csv_path), str(parquet_path), "zstd"))
    assert not res['ok']
    assert 'could not parse' in res['error'] or 'parsing' in res['error'] or 'error' in res['error'].lower()
    assert not os.path.exists(parquet_path)

# CLI argument parsing can be tested with monkeypatch
import sys
import argparse
from data import main as data_main

def test_cli_args(monkeypatch):
    test_args = ["data.py", "somefile.csv"]
    monkeypatch.setattr(sys, "argv", test_args)
    try:
        data_main()
    except SystemExit:
        pass  # argparse calls sys.exit 