import hashlib
import json
from pathlib import Path

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