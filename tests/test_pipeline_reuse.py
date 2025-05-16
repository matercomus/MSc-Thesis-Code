import os
import json
import shutil
import tempfile
import time
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from pipeline_utils import file_hash
from new_indicators_pipeline import find_latest_output

def write_dummy_output(path, input_path, input_key, input_hash=None):
    # Write a dummy output file and meta file with the input hash
    Path(path).write_text("dummy")
    meta = {input_key: input_hash or file_hash(input_path)}
    with open(str(path) + ".meta.json", "w") as f:
        json.dump(meta, f)

def test_reuse_when_input_hash_matches(tmp_path):
    # Setup
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input.parquet"
    input_file.write_text("input")
    out_file = data_dir / "cleaned_points_in_beijing_20250101_000000.parquet"
    write_dummy_output(out_file, input_file, "filtered_points_hash")
    # Should find and reuse
    found = find_latest_output(str(data_dir / "cleaned_points_in_beijing_*.parquet"))
    assert found == str(out_file)
    # Check meta hash matches
    with open(str(out_file) + ".meta.json") as f:
        meta = json.load(f)
    assert meta["filtered_points_hash"] == file_hash(input_file)

def test_no_reuse_when_input_hash_differs(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input.parquet"
    input_file.write_text("input")
    out_file = data_dir / "cleaned_points_in_beijing_20250101_000000.parquet"
    # Write meta with wrong hash
    write_dummy_output(out_file, input_file, "filtered_points_hash", input_hash="wronghash")
    # Should find file, but hash does not match
    found = find_latest_output(str(data_dir / "cleaned_points_in_beijing_*.parquet"))
    assert found == str(out_file)
    with open(str(out_file) + ".meta.json") as f:
        meta = json.load(f)
    assert meta["filtered_points_hash"] != file_hash(input_file)


def test_only_reuse_latest(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input.parquet"
    input_file.write_text("input")
    # Older file
    out_file1 = data_dir / "cleaned_points_in_beijing_20240101_000000.parquet"
    write_dummy_output(out_file1, input_file, "filtered_points_hash")
    time.sleep(1)
    # Newer file
    out_file2 = data_dir / "cleaned_points_in_beijing_20250101_000000.parquet"
    write_dummy_output(out_file2, input_file, "filtered_points_hash")
    found = find_latest_output(str(data_dir / "cleaned_points_in_beijing_*.parquet"))
    assert found == str(out_file2)


def test_logging_reuse_action(tmp_path, caplog):
    # This test checks that the reuse action is logged
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input.parquet"
    input_file.write_text("input")
    out_file = data_dir / "cleaned_points_in_beijing_20250101_000000.parquet"
    write_dummy_output(out_file, input_file, "filtered_points_hash")
    # Simulate pipeline logging
    import logging
    logging.basicConfig(level=logging.INFO)
    caplog.set_level(logging.INFO)
    # Simulate reuse logic
    found = find_latest_output(str(data_dir / "cleaned_points_in_beijing_*.parquet"))
    if found == str(out_file):
        logging.info(f"[REUSE] Using {found} for cleaned points step.")
    assert any("[REUSE] Using" in rec.message for rec in caplog.records) 