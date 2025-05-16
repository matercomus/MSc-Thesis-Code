import hashlib
import json
from pathlib import Path
import polars as pl
from itertools import combinations
from typing import List, Dict, Any, Optional

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def write_meta(meta_path, meta_dict):
    with open(meta_path, 'w') as f:
        json.dump(meta_dict, f)

def read_meta(meta_path):
    if not Path(meta_path).exists():
        return None
    with open(meta_path, 'r') as f:
        return json.load(f)

def is_up_to_date(output_path, input_paths):
    meta_path = str(output_path) + '.meta.json'
    if not Path(output_path).exists() or not Path(meta_path).exists():
        return False
    meta = read_meta(meta_path)
    for key, path in input_paths.items():
        if meta.get(key) != file_hash(path):
            return False
    return True

class PipelineStats:
    def __init__(self, run_id: str, output_dir: str = "pipeline_stats"):
        self.run_id = run_id
        self.output_dir = output_dir
        self.stats = {
            "run_id": run_id,
            "steps": {},
            "indicator_overlaps": {},
            "filtering": {},
            "meta": {},
        }

    def record_filtering(self, step: str, before: int, after: int, criteria: str, extra: Optional[Dict[str, Any]] = None):
        pct = 100.0 * (before - after) / before if before > 0 else 0.0
        self.stats["filtering"][step] = {
            "before": before,
            "after": after,
            "filtered": before - after,
            "filtered_pct": pct,
            "criteria": criteria,
        }
        if extra:
            self.stats["filtering"][step].update(extra)

    def record_step_stats(self, step: str, df: pl.DataFrame, abnormal_col: Optional[str] = None):
        n_total = df.height
        step_stats = {"n_total": n_total}
        if abnormal_col and abnormal_col in df.columns:
            n_abnormal = df.filter(pl.col(abnormal_col)).height
            step_stats["n_abnormal"] = n_abnormal
            step_stats["abnormal_pct"] = 100.0 * n_abnormal / n_total if n_total > 0 else 0.0
        self.stats["steps"][step] = step_stats

    def record_indicator_flags(self, df: pl.DataFrame, indicator_cols: List[str]):
        # For each indicator, count flagged
        for col in indicator_cols:
            if col in df.columns:
                n_flagged = df.filter(pl.col(col)).height
                self.stats["indicator_overlaps"][col] = {
                    "n_flagged": n_flagged,
                    "flagged_pct": 100.0 * n_flagged / df.height if df.height > 0 else 0.0,
                }
        # For all combinations
        for r in range(2, len(indicator_cols) + 1):
            for combo in combinations(indicator_cols, r):
                mask = pl.lit(True)
                for col in combo:
                    mask = mask & pl.col(col)
                n_combo = df.filter(mask).height
                key = " & ".join(combo)
                self.stats["indicator_overlaps"][key] = {
                    "n_flagged": n_combo,
                    "flagged_pct": 100.0 * n_combo / df.height if df.height > 0 else 0.0,
                }

    def record_meta(self, key: str, value: Any):
        self.stats["meta"][key] = value

    def save(self):
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, self.run_id, "pipeline_stats.json")
        with open(out_path, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)

    def get_stats(self):
        return self.stats 