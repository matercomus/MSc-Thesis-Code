# process_large_csvs.ps1
# Description: Copy large CSVs from Windows drive, convert to Parquet, log progress, and clean up.

$SRC_DIR = "I:\北京市出租车GPS数据"
$DEST_DIR = "C:\Users\matt\Dev\MSc-Thesis-Code\data"
$PARQUET_DIR = "$DEST_DIR\parquet"
$LOG_FILE = "$DEST_DIR\process_log.txt"
$PY_SCRIPT = "C:\Users\matt\Dev\MSc-Thesis-Code\data.py"

# Ensure directories exist
New-Item -ItemType Directory -Path $DEST_DIR -Force | Out-Null
New-Item -ItemType Directory -Path $PARQUET_DIR -Force | Out-Null

function Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMsg = "[$timestamp] $Message"
    $logMsg | Tee-Object -FilePath $LOG_FILE -Append
}

Log "==== Starting batch CSV to Parquet processing ===="

$startDate = Get-Date "2019-11-25"
$endDate = Get-Date "2019-12-01"

for ($date = $startDate; $date -le $endDate; $date = $date.AddDays(1)) {
    $dateStr = $date.ToString("yyyy.MM.dd")
    $csvFile = Join-Path $SRC_DIR "$dateStr.csv"
    if (Test-Path $csvFile) {
        $fname = Split-Path $csvFile -Leaf
        $parquetOut = Join-Path $PARQUET_DIR ($fname -replace '\.csv$', '.parquet')

        Log "--- Processing: $fname ---"
        $fileSize = (Get-Item $csvFile).Length / 1MB
        Log ("File size: {0:N2} MB" -f $fileSize)
        $startTime = Get-Date
        Log "Converting $csvFile to Parquet..."
        $pythonResult = & uv run python "$PY_SCRIPT" "$csvFile" --output-dir "$PARQUET_DIR" 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
        if ($LASTEXITCODE -eq 0) {
            $endTime = Get-Date
            $duration = ($endTime - $startTime).TotalSeconds
            Log "Conversion complete: $parquetOut (Duration: ${duration}s)"
            if (Test-Path $parquetOut) {
                $parquetSize = (Get-Item $parquetOut).Length / 1MB
                Log ("Parquet size: {0:N2} MB" -f $parquetSize)
            }
        }
        else {
            Log "ERROR: Conversion failed for $csvFile. Exiting script."
            exit 1
        }
        Log "--- Done with: $fname ---"
        Log ""
    }
    else {
        Log "File not found: $csvFile. Skipping."
    }
}

Log "==== All files processed ====" 