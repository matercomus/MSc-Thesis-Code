import polars as pl
import glob
from datetime import datetime
import sys

def check_csv_file(file_path: str, log_file) -> None:
    """Check schema and first line of a CSV file efficiently."""
    separator = "="*80
    output = [
        f"\n{separator}",
        f"Checking file: {file_path}",
        separator
    ]
    
    try:
        # Define schema with all columns as strings to avoid parsing errors
        schema = {
            'company': pl.String,
            'phone': pl.String,
            'plate': pl.String,
            'timestamp': pl.String,
            'longitude': pl.String,
            'latitude': pl.String,
            'speed': pl.String,
            'direction': pl.String,
            'status': pl.String,
            'angle': pl.String,
            'status_code': pl.String
        }
        
        # Read first row with string schema
        df = pl.read_csv(
            file_path,
            n_rows=1,
            has_header=False,
            encoding='utf8',
            schema=schema
        )
        
        output.append("\nFirst row:")
        for col, val in zip(df.columns, df.row(0)):
            output.append(f"  {col}: {val}")
            
    except Exception as e:
        output.append(f"Error reading file: {e}")
        
        # Try reading raw bytes as fallback
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(1000)
                output.append("\nFirst 1000 bytes:")
                output.append(str(raw))
        except Exception as e2:
            output.append(f"Error reading raw bytes: {e2}")
    
    # Write to both console and log file
    for line in output:
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

def main():
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"csv_check_{timestamp}.log"
    
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        # Write header
        header = f"CSV File Check Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(header)
        log_file.write(header + "\n")
        
        # Get all CSV files for the date range
        base_path = r"I:\北京市出租车GPS数据"
        files = sorted(glob.glob(f"{base_path}\\2019.11.2[5-9].csv") + 
                      glob.glob(f"{base_path}\\2019.11.30.csv") +
                      glob.glob(f"{base_path}\\2019.12.01.csv"))
        
        file_count = f"Found {len(files)} files to check"
        print(file_count)
        log_file.write(file_count + "\n")
        
        # Check each file
        for file_path in files:
            check_csv_file(file_path, log_file)

if __name__ == "__main__":
    main() 