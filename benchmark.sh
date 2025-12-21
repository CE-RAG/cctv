#!/bin/bash

# Configuration
API_URL="${API_URL:-http://192.168.248.177:8000/predict}"
INPUT_FILE="${1:-file_paths.txt}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found"
    exit 1
fi

# Initialize variables
total_time=0
count=0
success_count=0
failed_count=0

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           API Response Time Benchmark                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "API Endpoint: $API_URL"
echo "Input File:   $INPUT_FILE"
echo ""
echo "────────────────────────────────────────────────────────────────"

# Read and process each file path
while IFS= read -r file_path; do
    # Skip empty lines
    [[ -z "$file_path" ]] && continue
    
    count=$((count + 1))
    
    # Truncate long paths for display
    display_path="$file_path"
    if [ ${#display_path} -gt 50 ]; then
        display_path="...${display_path: -47}"
    fi
    
    printf "[%3d] %-50s " "$count" "$display_path"
    
    # Measure time and make API call
    start_time=$(date +%s.%N)
    
    http_code=$(curl -s -w "%{http_code}" -o /dev/null \
                     -X POST "$API_URL" \
                     -H "Content-Type: application/json" \
                     -d "{\"image_path\": \"$file_path\"}")
    
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    total_time=$(echo "$total_time + $elapsed" | bc)
    
    # Display result
    if [ "$http_code" -eq 200 ]; then
        success_count=$((success_count + 1))
        printf "✓ %6.3fs\n" "$elapsed"
    else
        failed_count=$((failed_count + 1))
        printf "✗ %6.3fs (HTTP %s)\n" "$elapsed" "$http_code"
    fi
    
done < "$INPUT_FILE"

echo "────────────────────────────────────────────────────────────────"
echo ""

# Calculate statistics
if [ $count -gt 0 ]; then
    average_time=$(echo "scale=3; $total_time / $count" | bc)
    
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                         SUMMARY                                ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    printf "║  Total Requests:        %-35d║\n" "$count"
    printf "║  Successful:            %-35d║\n" "$success_count"
    printf "║  Failed:                %-35d║\n" "$failed_count"
    echo "╠════════════════════════════════════════════════════════════════╣"
    printf "║  Overall Time:          %-31.3fs    ║\n" "$total_time"
    printf "║  Average Time:          %-31.3fs    ║\n" "$average_time"
    
    if [ $success_count -gt 0 ]; then
        avg_success=$(echo "scale=3; $total_time / $success_count" | bc)
        printf "║  Avg (Success Only):    %-31.3fs    ║\n" "$avg_success"
    fi
    
    # Calculate requests per second
    if [ $(echo "$total_time > 0" | bc) -eq 1 ]; then
        rps=$(echo "scale=2; $count / $total_time" | bc)
        printf "║  Requests/Second:       %-31.2f    ║\n" "$rps"
    fi
    
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
else
    echo "No files processed"
fi