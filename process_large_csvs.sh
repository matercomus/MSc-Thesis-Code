#!/bin/bash

# Script: process_large_csvs.sh
# Description: Copy large CSVs from Windows drive to WSL2, convert to Parquet, log progress, and clean up.
# Usage: bash process_large_csvs.sh

set -euo pipefail

# Configurable paths
SRC_DIR="/mnt/i/北京市出租车GPS数据"
DEST_DIR="$HOME/MSc-Thesis-Code/data"
PARQUET_DIR="$DEST_DIR/parquet"
LOG_FILE="$DEST_DIR/process_log.txt"
PY_SCRIPT="$HOME/MSc-Thesis-Code/data.py"

mkdir -p "$DEST_DIR" "$PARQUET_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "==== Starting batch CSV to Parquet processing ===="

for f in "$SRC_DIR"/*_区域内.csv; do
    fname=$(basename "$f")
    dest_csv="$DEST_DIR/$fname"
    parquet_out="$PARQUET_DIR/${fname%.csv}.parquet"

    log "--- Processing: $fname ---"
    log "Copying $f to $dest_csv using rsync..."
    if rsync -ah --progress "$f" "$dest_csv" 2>&1 | tee -a "$LOG_FILE"; then
        log "Copy complete: $dest_csv"
    else
        log "ERROR: Failed to copy $f. Skipping."
        continue
    fi

    log "File size: $(du -h "$dest_csv" | cut -f1)"
    start_time=$(date +%s)
    log "Converting $dest_csv to Parquet..."
    if python3 "$PY_SCRIPT" "$dest_csv" --output-dir "$PARQUET_DIR" 2>&1 | tee -a "$LOG_FILE"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log "Conversion complete: $parquet_out (Duration: ${duration}s)"
        log "Parquet size: $(du -h "$parquet_out" | cut -f1)"
        rm -f "$dest_csv"
        log "Deleted $dest_csv to save space."
    else
        log "ERROR: Conversion failed for $dest_csv. Keeping file for inspection."
    fi
    log "--- Done with: $fname ---"
    log ""
done

log "==== All files processed ====" 