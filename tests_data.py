import pytest
import os
from pathlib import Path
from data import get_system_resources, find_csv_files, get_file_size, convert_one_file

def test_get_system_resources():
    n_cores, mem_gb = get_system_resources()
    assert isinstance(n_cores, int)
    assert n_cores > 0
    assert isinstance(mem_gb, float)
    assert mem_gb > 0

def test_find_csv_files(tmp_path):
    # Create some fake csv files
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    f1.write_text("test")
    f2.write_text("test")
    files = find_csv_files([str(tmp_path)])
    assert str(f1) in files
    assert str(f2) in files
    # Test glob
    files2 = find_csv_files([str(tmp_path / "*.csv")])
    assert set(files) == set(files2)

def test_get_file_size(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello world")
    size = get_file_size(str(f))
    assert size == len("hello world")
    assert get_file_size("nonexistent.txt") == 0

def test_convert_one_file_error(tmp_path):
    # Should handle missing file gracefully
    args = (str(tmp_path / "nofile.csv"), str(tmp_path / "out.parquet"), "zstd")
    res = convert_one_file(args)
    assert not res['ok']
    assert 'error' in res

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